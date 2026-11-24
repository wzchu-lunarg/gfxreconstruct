/*
** Copyright (c) 2026 LunarG, Inc.
**
** Permission is hereby granted, free of charge, to any person obtaining a
** copy of this software and associated documentation files (the "Software"),
** to deal in the Software without restriction, including without limitation
** the rights to use, copy, modify, merge, publish, distribute, sublicense,
** and/or sell copies of the Software, and to permit persons to whom the
** Software is furnished to do so, subject to the following conditions:
**
** The above copyright notice and this permission notice shall be included in
** all copies or substantial portions of the Software.
**
** THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
** IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
** FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
** AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
** LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
** FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
** DEALINGS IN THE SOFTWARE.
*/

#include "decode/vulkan_stage_frame_loop_image_restore.h"

#include "decode/vulkan_temporary_objects.h"
#include "generated/generated_vulkan_enum_to_string.h"
#include "graphics/vulkan_device_util.h"
#include "graphics/vulkan_image_layout_map_util.h"
#include "graphics/vulkan_resources_util.h"
#include "graphics/vulkan_struct_get_pnext.h"
#include "graphics/vulkan_util.h"
#include "util/callbacks.h"
#include "util/logging.h"

#include <cinttypes>
#include <cstdlib>
#include <limits>

GFXRECON_BEGIN_NAMESPACE(gfxrecon)
GFXRECON_BEGIN_NAMESPACE(decode)

bool IsFrameEndBoundary(const PNextNode* pnext)
{
    const Decoded_VkFrameBoundaryEXT* frame_boundary = GetPNextMetaStruct<Decoded_VkFrameBoundaryEXT>(pnext);
    return (frame_boundary != nullptr) && (frame_boundary->decoded_value != nullptr) &&
           ((frame_boundary->decoded_value->flags & VK_FRAME_BOUNDARY_FRAME_END_BIT_EXT) != 0);
}

#define CHECK_VK_RESULT(_res_, _func_)                                        \
    {                                                                         \
        VkResult _RES = (_res_);                                              \
        if (_RES != VK_SUCCESS)                                               \
        {                                                                     \
            GFXRECON_LOG_ERROR("[%s:%u] synthetic call to %s failed with %s", \
                               __FILE__,                                      \
                               __LINE__,                                      \
                               _func_,                                        \
                               util::ToString(_RES).c_str());                 \
            std::exit(-1);                                                    \
        }                                                                     \
    }

static void CmdImageLayoutBarriers(const graphics::VulkanDeviceTable&       device_table,
                                   VkCommandBuffer                          command_buffer,
                                   const std::vector<VkImageMemoryBarrier>& barriers)
{
    if (barriers.empty())
    {
        return;
    }

    device_table.CmdPipelineBarrier(command_buffer,
                                    VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                    VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                    0,
                                    0,
                                    nullptr,
                                    0,
                                    nullptr,
                                    static_cast<uint32_t>(barriers.size()),
                                    barriers.data());
}

VulkanStageFrameLoopImageRestore::ImageTracking&
VulkanStageFrameLoopImageRestore::GetImageTracking(format::HandleId device)
{
    auto it = per_device_image_tracking_.find(device);
    if (it == per_device_image_tracking_.end())
    {
        auto&             object_table = object_info_table_;
        VulkanDeviceInfo* device_info  = object_table.GetVkDeviceInfo(device);
        GFXRECON_ASSERT(device_info != nullptr);
        const auto& device_table = *GetDeviceTable(device_info->handle);

        VulkanPhysicalDeviceInfo* phys_info = object_table.GetVkPhysicalDeviceInfo(device_info->parent_id);
        GFXRECON_ASSERT(phys_info != nullptr);
        const VkPhysicalDeviceMemoryProperties* memory_properties = &phys_info->capture_memory_properties;
        if (phys_info->replay_device_info != nullptr && phys_info->replay_device_info->memory_properties.has_value())
        {
            memory_properties = &phys_info->replay_device_info->memory_properties.value();
        }

        auto result = per_device_image_tracking_.emplace(
            device, ImageTracking(device, device_table, object_table, device_info->allocator, memory_properties));
        it = result.first;
        GFXRECON_ASSERT(result.second);
    }
    return it->second;
}

bool VulkanStageFrameLoopImageRestore::CanSnapshotImageContents(const VulkanImageInfo* image_info) const
{
    const format::HandleId image_id = image_info->capture_id;

    if (!restorable_images_.contains(image_id))
    {
        return false;
    }

    if (image_info->memory_property_flags == 0)
    {
        GFXRECON_LOG_DEBUG("TrackImageStates: Skipping image %" PRIu64
                           " with no bound memory; its contents will not be restored across loop repetitions.",
                           image_id);
        return false;
    }

    if ((image_info->memory_property_flags & VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT) != 0)
    {
        GFXRECON_LOG_DEBUG("TrackImageStates: Skipping image %" PRIu64
                           " backed by lazily allocated memory; its contents will not be restored across loop "
                           "repetitions.",
                           image_id);
        return false;
    }

    return true;
}

void VulkanStageFrameLoopImageRestore::TrackImageStates()
{
    // Images that exists now and therefore needs its layouts recorded.
    std::unordered_map<format::HandleId, std::vector<format::HandleId>> device_images;
    // Images that can have content restored.
    std::unordered_map<format::HandleId, std::vector<format::HandleId>> device_restorable_images;

    object_info_table_.VisitVkImageInfo(
        [this, &device_images, &device_restorable_images](const VulkanImageInfo* image_info) {
            if (image_info->handle == VK_NULL_HANDLE)
            {
                return;
            }

            // Table still holds stale entries after a `vkDestroySwapchainKHR`.
            if (image_info->swapchain_id != format::kNullHandleId)
            {
                return;
            }

            // An image should never outlive the device that created it in the object info table.
            GFXRECON_ASSERT(object_info_table_.GetVkDeviceInfo(image_info->parent_id) != nullptr);

            device_images[image_info->parent_id].push_back(image_info->capture_id);

            if (CanSnapshotImageContents(image_info))
            {
                device_restorable_images[image_info->parent_id].push_back(image_info->capture_id);
            }
        });

    for (const auto& [device_id, image_ids] : device_images)
    {
        GetImageTracking(device_id).RecordInitialState(image_ids, device_restorable_images[device_id]);
    }
}

void VulkanStageFrameLoopImageRestore::FixupDeviceImages(format::HandleId device, format::HandleId queue)
{
    auto it = per_device_image_tracking_.find(device);
    if (it == per_device_image_tracking_.end())
    {
        return;
    }

    it->second.Restore(queue);
}

void VulkanStageFrameLoopImageRestore::ResetImageTracking()
{
    // Destroying the shadow resources of the loop is not a call of the application.
    util::MarkInjectedCommandsHelper injected_commands_scope;

    for (auto& [device_id, image_tracking] : per_device_image_tracking_)
    {
        image_tracking.DestroyShadowImages();
    }
    per_device_image_tracking_.clear();
}

void VulkanStageFrameLoopImageRestore::ResetImageTracking(format::HandleId device)
{
    // Destroying the shadow resources of the loop is not a call of the application.
    util::MarkInjectedCommandsHelper injected_commands_scope;

    auto it = per_device_image_tracking_.find(device);
    if (it != per_device_image_tracking_.end())
    {
        it->second.DestroyShadowImages();
        per_device_image_tracking_.erase(it);
    }
}

// Creates the device-local image that an image's contents are snapshotted into, and binds memory to it.
bool VulkanStageFrameLoopImageRestore::ImageTracking::CreateShadowImage(format::HandleId       image_id,
                                                                        const VulkanImageInfo* image_info,
                                                                        ImageState&            state)
{
    VulkanDeviceInfo* device_info = object_table_.GetVkDeviceInfo(device_id_);
    GFXRECON_ASSERT(device_info != nullptr);

    VkImageCreateInfo create_info = { VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
    create_info.imageType         = image_info->type;
    create_info.format            = image_info->format;
    create_info.extent            = image_info->extent;
    create_info.mipLevels         = image_info->level_count;
    create_info.arrayLayers       = image_info->layer_count;
    create_info.samples           = image_info->sample_count;
    create_info.tiling            = VK_IMAGE_TILING_OPTIMAL;
    create_info.usage             = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    create_info.sharingMode       = VK_SHARING_MODE_EXCLUSIVE;
    create_info.initialLayout     = VK_IMAGE_LAYOUT_UNDEFINED;

    VkResult result = allocator_->CreateImageDirect(&create_info, nullptr, &state.shadow_image, &state.alloc_data);
    if (result != VK_SUCCESS)
    {
        GFXRECON_LOG_WARNING("Failed to create shadow image for image %" PRIu64
                             " with %s; its contents will not be restored across loop repetitions.",
                             image_id,
                             util::ToString(result).c_str());
        state.shadow_image = VK_NULL_HANDLE;
        return false;
    }

    VkMemoryRequirements mem_reqs;
    device_table_.GetImageMemoryRequirements(device_info->handle, state.shadow_image, &mem_reqs);

    uint32_t memory_type_index =
        graphics::GetMemoryTypeIndex(*memory_properties_, mem_reqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (memory_type_index == std::numeric_limits<uint32_t>::max())
    {
        memory_type_index = graphics::GetMemoryTypeIndex(*memory_properties_, mem_reqs.memoryTypeBits, 0);
    }
    if (memory_type_index == std::numeric_limits<uint32_t>::max())
    {
        GFXRECON_LOG_WARNING("No suitable memory type for shadow image for image %" PRIu64
                             "; its contents will not be restored across loop repetitions.",
                             image_id);
        return false;
    }

    VkMemoryAllocateInfo alloc_info = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    alloc_info.allocationSize       = mem_reqs.size;
    alloc_info.memoryTypeIndex      = memory_type_index;

    result = allocator_->AllocateMemoryDirect(&alloc_info, nullptr, &state.shadow_memory, &state.mem_data);
    if (result != VK_SUCCESS)
    {
        GFXRECON_LOG_WARNING("Failed to allocate shadow memory for image %" PRIu64 " (size %" PRIu64
                             ") with %s; its contents will not be restored across loop repetitions.",
                             image_id,
                             mem_reqs.size,
                             util::ToString(result).c_str());
        state.shadow_memory = VK_NULL_HANDLE;
        return false;
    }

    VkMemoryPropertyFlags bind_properties = 0;
    result                                = allocator_->BindImageMemoryDirect(
        state.shadow_image, state.shadow_memory, 0, state.alloc_data, state.mem_data, &bind_properties);
    if (result != VK_SUCCESS)
    {
        GFXRECON_LOG_WARNING("Failed to bind shadow memory for image %" PRIu64
                             " with %s; its contents will not be restored across loop repetitions.",
                             image_id,
                             util::ToString(result).c_str());
        return false;
    }

    const VkImageAspectFlags aspects         = state.initial_layouts.GetAspects();
    const VkImageAspectFlags barrier_aspects = graphics::ToBarrierAspects(aspects, aspects);
    state.shadow_range = { barrier_aspects, 0, image_info->level_count, 0, image_info->layer_count };

    return true;
}

void VulkanStageFrameLoopImageRestore::ImageTracking::DestroyShadowImage(ImageState& state)
{
    if (allocator_ != nullptr)
    {
        if (state.shadow_image != VK_NULL_HANDLE)
        {
            allocator_->DestroyImageDirect(state.shadow_image, nullptr, state.alloc_data);
        }
        if (state.shadow_memory != VK_NULL_HANDLE)
        {
            allocator_->FreeMemoryDirect(state.shadow_memory, nullptr, state.mem_data);
        }
    }

    state.shadow_image  = VK_NULL_HANDLE;
    state.shadow_memory = VK_NULL_HANDLE;
    state.copyable_ranges.clear();
    state.excluded_ranges.clear();
    state.copy_regions.clear();
}

void VulkanStageFrameLoopImageRestore::ImageTracking::RecordInitialState(
    const std::vector<format::HandleId>& image_ids, const std::vector<format::HandleId>& restorable_image_ids)
{
    restore_commands_ = {};

    // Record layouts
    for (format::HandleId image_id : image_ids)
    {
        const VulkanImageInfo* image_info = object_table_.GetVkImageInfo(image_id);
        if (image_info == nullptr)
        {
            continue;
        }

        image_states_[image_id].initial_layouts = image_info->subresource_layouts;
    }

    if (allocator_ == nullptr || restorable_image_ids.empty())
    {
        return;
    }

    VulkanDeviceInfo* device_info = object_table_.GetVkDeviceInfo(device_id_);
    GFXRECON_ASSERT(device_info != nullptr);

    TemporaryCommandBuffer temp_cmd_buff(*device_info, device_table_);
    if (temp_cmd_buff.CreateAndBegin(graphics::FindGraphicsOrComputeQueueFamilyIndex) != VK_SUCCESS)
    {
        return;
    }

    std::vector<format::HandleId> recorded_ids;

    for (format::HandleId image_id : restorable_image_ids)
    {
        const VulkanImageInfo* image_info = object_table_.GetVkImageInfo(image_id);
        if (image_info == nullptr || image_info->handle == VK_NULL_HANDLE)
        {
            continue;
        }

        ImageState& state = image_states_[image_id];

        graphics::BuildCopyableSubresourceRanges(image_info->format,
                                                 image_info->level_count,
                                                 image_info->layer_count,
                                                 state.initial_layouts,
                                                 state.copyable_ranges,
                                                 state.excluded_ranges);

        if (state.copyable_ranges.empty())
        {
            // Every subresource started out undefined and have nothing to be restored to.
            GFXRECON_LOG_DEBUG("RecordInitialState: Skipping image %" PRIu64
                               " with no defined subresource layouts; its contents will not be restored across "
                               "loop repetitions.",
                               image_id);
            state.excluded_ranges.clear();
            continue;
        }

        graphics::BuildImageCopyRegions(
            image_info->format, image_info->extent, state.copyable_ranges, state.copy_regions);

        if (!CreateShadowImage(image_id, image_info, state))
        {
            DestroyShadowImage(state);
            continue;
        }

        recorded_ids.push_back(image_id);
    }

    if (recorded_ids.empty())
    {
        return;
    }

    // Puts every copyable subresource of each source image into VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL.
    std::vector<VkImageMemoryBarrier> pre_barriers;

    for (format::HandleId image_id : recorded_ids)
    {
        const VulkanImageInfo* image_info = object_table_.GetVkImageInfo(image_id);
        const ImageState&      state      = image_states_.at(image_id);

        graphics::AppendTransferLayoutBarriers(image_info->handle,
                                               image_info->subresource_layouts,
                                               state.excluded_ranges,
                                               VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                               pre_barriers);
    }

    // Transitions those subresources back out of VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL.
    std::vector<VkImageMemoryBarrier> post_barriers = pre_barriers;

    graphics::ReverseLayoutBarriers(post_barriers);

    // Transition shadows
    for (format::HandleId image_id : recorded_ids)
    {
        const ImageState& state = image_states_.at(image_id);

        pre_barriers.push_back(graphics::MakeLayoutRestoreBarrier(
            state.shadow_image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, state.shadow_range));
        post_barriers.push_back(graphics::MakeLayoutRestoreBarrier(state.shadow_image,
                                                                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                                                   VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                                                   state.shadow_range));
    }

    // Transition images to VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL and shadows to VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL.
    CmdImageLayoutBarriers(device_table_, temp_cmd_buff.command_buffer, pre_barriers);

    // Copy the contents of each source image into its shadow.
    for (format::HandleId image_id : recorded_ids)
    {
        const VulkanImageInfo* image_info = object_table_.GetVkImageInfo(image_id);
        const ImageState&      state      = image_states_.at(image_id);

        device_table_.CmdCopyImage(temp_cmd_buff.command_buffer,
                                   image_info->handle,
                                   VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                   state.shadow_image,
                                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                   static_cast<uint32_t>(state.copy_regions.size()),
                                   state.copy_regions.data());
    }

    // Transition images back to their initial layouts and shadows to VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL.
    CmdImageLayoutBarriers(device_table_, temp_cmd_buff.command_buffer, post_barriers);

    CHECK_VK_RESULT(temp_cmd_buff.SubmitAndDestroy(), "TemporaryCommandBuffer::SubmitAndDestroy");
}

void VulkanStageFrameLoopImageRestore::ImageTracking::BuildRestoreCommands()
{
    RestoreCommands& commands = restore_commands_;
    commands.built            = true;

    for (auto& [image_id, state] : image_states_)
    {
        VulkanImageInfo* image_info = object_table_.GetVkImageInfo(image_id);
        if (image_info == nullptr || image_info->handle == VK_NULL_HANDLE)
        {
            continue;
        }

        const size_t first_post_barrier = commands.post_barriers.size();

        if (state.HasShadow())
        {
            graphics::AppendTransferLayoutBarriers(image_info->handle,
                                                   image_info->subresource_layouts,
                                                   state.excluded_ranges,
                                                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                                   commands.pre_barriers);

            image_info->subresource_layouts = state.initial_layouts;
            graphics::AppendTransferLayoutBarriers(image_info->handle,
                                                   image_info->subresource_layouts,
                                                   state.excluded_ranges,
                                                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                                   commands.post_barriers);
            graphics::ReverseLayoutBarriers(commands.post_barriers, first_post_barrier);

            commands.copy_ids.push_back(image_id);
        }
        else // Image contents could not be restored but still restore the layouts to their initial state.
        {
            graphics::AppendImageLayoutBarriers(
                image_info->handle, image_info->subresource_layouts, state.initial_layouts, commands.post_barriers);
            if (commands.post_barriers.size() == first_post_barrier)
            {
                continue;
            }

            image_info->subresource_layouts = state.initial_layouts;
        }

        const VkImageLayout intermediate_layout = (commands.post_barriers.size() > first_post_barrier)
                                                      ? commands.post_barriers.back().newLayout
                                                      : VK_IMAGE_LAYOUT_MAX_ENUM;
        if (intermediate_layout != VK_IMAGE_LAYOUT_MAX_ENUM)
        {
            image_info->intermediate_layout = intermediate_layout;
        }

        commands.layout_updates.emplace_back(image_id, intermediate_layout);
    }
}

void VulkanStageFrameLoopImageRestore::ImageTracking::Restore(format::HandleId queue)
{
    if (image_states_.empty())
    {
        return;
    }

    VulkanDeviceInfo* device_info = object_table_.GetVkDeviceInfo(device_id_);
    VulkanQueueInfo*  queue_info  = object_table_.GetVkQueueInfo(queue);
    GFXRECON_ASSERT(device_info != nullptr && queue_info != nullptr);

    const bool built_this_restore = !restore_commands_.built;
    if (built_this_restore)
    {
        BuildRestoreCommands();
    }
    const RestoreCommands& commands = restore_commands_;

    if (commands.copy_ids.empty() && commands.post_barriers.empty())
    {
        return;
    }

    if (!built_this_restore)
    {
        for (const auto& [image_id, intermediate_layout] : commands.layout_updates)
        {
            VulkanImageInfo* image_info = object_table_.GetVkImageInfo(image_id);
            GFXRECON_ASSERT(image_info != nullptr);

            image_info->subresource_layouts = image_states_.at(image_id).initial_layouts;
            if (intermediate_layout != VK_IMAGE_LAYOUT_MAX_ENUM)
            {
                image_info->intermediate_layout = intermediate_layout;
            }
        }
    }

    TemporaryCommandBuffer temp_cmd_buff(*device_info, device_table_);
    if (temp_cmd_buff.CreateAndBegin(queue_info->family_index, queue_info->handle) != VK_SUCCESS)
    {
        return;
    }

    // Transition images to VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL and shadows to
    // VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL.
    CmdImageLayoutBarriers(device_table_, temp_cmd_buff.command_buffer, commands.pre_barriers);

    // Copy the contents of each shadow into its source image.
    for (format::HandleId image_id : commands.copy_ids)
    {
        const VulkanImageInfo* image_info = object_table_.GetVkImageInfo(image_id);
        const ImageState&      state      = image_states_.at(image_id);

        device_table_.CmdCopyImage(temp_cmd_buff.command_buffer,
                                   state.shadow_image,
                                   VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                   image_info->handle,
                                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                   static_cast<uint32_t>(state.copy_regions.size()),
                                   state.copy_regions.data());
    }

    // Transition images to their initial layouts and shadows to VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL.
    CmdImageLayoutBarriers(device_table_, temp_cmd_buff.command_buffer, commands.post_barriers);

    CHECK_VK_RESULT(temp_cmd_buff.SubmitAndDestroy(), "TemporaryCommandBuffer::SubmitAndDestroy");
}

void VulkanStageFrameLoopImageRestore::ImageTracking::DestroyShadowImages()
{
    for (auto& [image_id, state] : image_states_)
    {
        DestroyShadowImage(state);
    }
    image_states_.clear();
    restore_commands_ = {};
}

// Stage

VulkanStageFrameLoopImageRestore::VulkanStageFrameLoopImageRestore(CommonObjectInfoTable&    object_info_table,
                                                                   graphics::FrameLoopInfo&  frame_loop_info,
                                                                   VulkanDispatchTableLookup dispatch_tables) :
    object_info_table_(object_info_table),
    frame_loop_info_(frame_loop_info), dispatch_tables_(std::move(dispatch_tables))
{
    GFXRECON_ASSERT(dispatch_tables_.get_instance_table && dispatch_tables_.get_device_table);
}

VulkanStageFrameLoopImageRestore::~VulkanStageFrameLoopImageRestore()
{
    ResetImageTracking();
}

const graphics::VulkanInstanceTable* VulkanStageFrameLoopImageRestore::GetInstanceTable(const void* handle) const
{
    return dispatch_tables_.get_instance_table(handle);
}

const graphics::VulkanDeviceTable* VulkanStageFrameLoopImageRestore::GetDeviceTable(const void* handle) const
{
    return dispatch_tables_.get_device_table(handle);
}

void VulkanStageFrameLoopImageRestore::RecordLoopStartState()
{
    // The calls that save the state of the loop are not calls of the application.
    util::MarkInjectedCommandsHelper injected_commands_scope;

    // The replay consumer waits for every device it created.
    VulkanConsumerStage::WaitDevicesIdle();

    GFXRECON_LOG_DEBUG("VulkanStageFrameLoopImageRestore: recording the image state at the start of the loop");
    TrackImageStates();
}

void VulkanStageFrameLoopImageRestore::EndFrame(format::HandleId queue)
{
    if (frame_loop_info_.IsFinalIteration())
    {
        return;
    }

    const VulkanQueueInfo* queue_info = object_info_table_.GetVkQueueInfo(queue);
    if (queue_info == nullptr)
    {
        return;
    }

    VkDevice device = queue_info->parent;
    GFXRECON_ASSERT(device != VK_NULL_HANDLE);
    const graphics::VulkanDeviceTable* device_table = GetDeviceTable(device);
    GFXRECON_ASSERT(device_table != nullptr);

    // The calls that restore the state of the loop are not calls of the application.
    util::MarkInjectedCommandsHelper injected_commands_scope;

    GFXRECON_LOG_DEBUG("Waiting for device to idle...");
    CHECK_VK_RESULT(device_table->DeviceWaitIdle(device), "vkDeviceWaitIdle");

    FixupDeviceImages(queue_info->parent_id, queue);
}

// Loop start

void VulkanStageFrameLoopImageRestore::StartLooping()
{
    VulkanConsumerStage::StartLooping();
    RecordLoopStartState();
}

void VulkanStageFrameLoopImageRestore::ProcessStateEndMarker(uint64_t frame_number)
{
    VulkanConsumerStage::ProcessStateEndMarker(frame_number);

    // The loop started before the trim state finished loading, so the state that it recorded is out of date.
    if (frame_loop_info_.IsLooping())
    {
        ResetImageTracking();
        RecordLoopStartState();
    }
}

// Devices and images

void VulkanStageFrameLoopImageRestore::Process_vkDestroyDevice(const ApiCallInfo& call_info, args::DestroyDevice& args)
{
    // The replay consumer destroys a device that the loop uses only in the final iteration.
    if (!frame_loop_info_.IsLooping() || frame_loop_info_.IsFinalIteration())
    {
        ResetImageTracking(args.device);
    }

    Forward(call_info, args);
}

static bool CanRestoreImage(const VkImageCreateInfo* create_info)
{
    if (const auto* usage_flags2 = graphics::vulkan_struct_get_pnext<VkImageUsageFlags2CreateInfoKHR>(create_info))
    {
        if ((usage_flags2->usage & VK_IMAGE_USAGE_2_TRANSIENT_ATTACHMENT_BIT_KHR) != 0)
        {
            return false;
        }
    }
    else if ((create_info->usage & VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT) != 0)
    {
        return false;
    }

    if (const auto* stencil_usage2 = graphics::vulkan_struct_get_pnext<VkImageStencilUsage2CreateInfoKHR>(create_info))
    {
        if ((stencil_usage2->stencilUsage & VK_IMAGE_USAGE_2_TRANSIENT_ATTACHMENT_BIT_KHR) != 0)
        {
            return false;
        }
    }
    else if (const auto* stencil_usage = graphics::vulkan_struct_get_pnext<VkImageStencilUsageCreateInfo>(create_info))
    {
        if ((stencil_usage->stencilUsage & VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT) != 0)
        {
            return false;
        }
    }

    const auto* external_format = graphics::vulkan_struct_get_pnext<VkExternalFormatANDROID>(create_info);
    if ((external_format != nullptr) && (external_format->externalFormat != 0))
    {
        return false;
    }

    const auto* external_memory = graphics::vulkan_struct_get_pnext<VkExternalMemoryImageCreateInfo>(create_info);
    if ((external_memory != nullptr) &&
        ((external_memory->handleTypes & VK_EXTERNAL_MEMORY_HANDLE_TYPE_ANDROID_HARDWARE_BUFFER_BIT_ANDROID) != 0))
    {
        return false;
    }

    return true;
}

void VulkanStageFrameLoopImageRestore::Process_vkCreateImage(const ApiCallInfo& call_info, args::CreateImage& args)
{
    VkImageCreateInfo*      create_info = args.pCreateInfo.GetPointer();
    const VulkanDeviceInfo* device_info = object_info_table_.GetVkDeviceInfo(args.device);

    if ((create_info != nullptr) && !args.pImage.IsNull() && (device_info != nullptr) && CanRestoreImage(create_info) &&
        graphics::VulkanResourcesUtil::IsFormatSupported(*GetInstanceTable(device_info->parent),
                                                         device_info->parent,
                                                         create_info->format,
                                                         create_info->tiling,
                                                         VK_FORMAT_FEATURE_TRANSFER_SRC_BIT |
                                                             VK_FORMAT_FEATURE_TRANSFER_DST_BIT,
                                                         create_info))
    {
        create_info->usage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;

        if (auto* usage_flags2 = graphics::vulkan_struct_get_pnext<VkImageUsageFlags2CreateInfoKHR>(create_info))
        {
            usage_flags2->usage |= VK_IMAGE_USAGE_2_TRANSFER_SRC_BIT_KHR | VK_IMAGE_USAGE_2_TRANSFER_DST_BIT_KHR;
        }
        if (auto* stencil_usage = graphics::vulkan_struct_get_pnext<VkImageStencilUsageCreateInfo>(create_info))
        {
            stencil_usage->stencilUsage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        }
        if (auto* stencil_usage2 = graphics::vulkan_struct_get_pnext<VkImageStencilUsage2CreateInfoKHR>(create_info))
        {
            stencil_usage2->stencilUsage |=
                VK_IMAGE_USAGE_2_TRANSFER_SRC_BIT_KHR | VK_IMAGE_USAGE_2_TRANSFER_DST_BIT_KHR;
        }

        restorable_images_.insert(*args.pImage.GetPointer());
    }

    Forward(call_info, args);
}

// Submits and presents that end a frame

template <typename ArgsT>
void VulkanStageFrameLoopImageRestore::ProcessQueueSubmit(const ApiCallInfo& call_info, ArgsT& args)
{
    Forward(call_info, args);

    if (!frame_loop_info_.IsLooping())
    {
        return;
    }

    for (const auto& submit : args.pSubmits.GetMetaStructSpan())
    {
        if (IsFrameEndBoundary(submit.pNext))
        {
            EndFrame(args.queue);
        }
    }
}

void VulkanStageFrameLoopImageRestore::Process_vkQueueSubmit(const ApiCallInfo& call_info, args::QueueSubmit& args)
{
    ProcessQueueSubmit(call_info, args);
}

void VulkanStageFrameLoopImageRestore::Process_vkQueueSubmit2(const ApiCallInfo& call_info, args::QueueSubmit2& args)
{
    ProcessQueueSubmit(call_info, args);
}

void VulkanStageFrameLoopImageRestore::Process_vkQueueSubmit2KHR(const ApiCallInfo&     call_info,
                                                                 args::QueueSubmit2KHR& args)
{
    ProcessQueueSubmit(call_info, args);
}

void VulkanStageFrameLoopImageRestore::Process_vkQueueBindSparse(const ApiCallInfo&     call_info,
                                                                 args::QueueBindSparse& args)
{
    Forward(call_info, args);

    if (frame_loop_info_.IsLooping())
    {
        for (const Decoded_VkBindSparseInfo& bind_info : args.pBindInfo.GetMetaStructSpan())
        {
            if (IsFrameEndBoundary(bind_info.pNext))
            {
                EndFrame(args.queue);
            }
        }
    }
}

void VulkanStageFrameLoopImageRestore::Process_vkQueuePresentKHR(const ApiCallInfo&     call_info,
                                                                 args::QueuePresentKHR& args)
{
    Forward(call_info, args);

    if (frame_loop_info_.IsLooping())
    {
        EndFrame(args.queue);
    }
}

GFXRECON_END_NAMESPACE(decode)
GFXRECON_END_NAMESPACE(gfxrecon)

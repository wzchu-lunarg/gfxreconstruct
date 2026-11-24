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

#ifndef GFXRECON_DECODE_VULKAN_STAGE_FRAME_LOOP_IMAGE_RESTORE_H
#define GFXRECON_DECODE_VULKAN_STAGE_FRAME_LOOP_IMAGE_RESTORE_H

#include "decode/common_object_info_table.h"
#include "decode/vulkan_pnext_node.h"
#include "decode/vulkan_resource_allocator.h"
#include "generated/generated_vulkan_consumer_stage.h"
#include "generated/generated_vulkan_dispatch_table.h"
#include "graphics/frame_loop_info.h"
#include "graphics/vulkan_image_layout_map.h"

#include <functional>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

GFXRECON_BEGIN_NAMESPACE(gfxrecon)
GFXRECON_BEGIN_NAMESPACE(decode)

/// The dispatch tables that the replay consumer made for the instances and devices it created. The stage calls Vulkan
/// directly through them.
struct VulkanDispatchTableLookup
{
    std::function<const graphics::VulkanInstanceTable*(const void* handle)> get_instance_table;
    std::function<const graphics::VulkanDeviceTable*(const void* handle)>   get_device_table;
};

/// Returns true when a VkFrameBoundaryEXT in the chain ends a frame.
bool IsFrameEndBoundary(const PNextNode* pnext);

/// This stage puts images back in the state they were in when the frame loop started, before each repetition of the
/// loop: their contents come back from shadow copies, and their layouts go back to the layouts they started in.
///
/// The stage records the state when the loop starts, and again when trim state finishes loading while the loop runs.
/// It restores the state after the replay consumer handles a present or a submit that ends a frame. It reads the
/// handles and the image layouts from the object info table of the replay consumer, so the image layout stage has to
/// come after it.
///
/// To copy images, the stage adds the transfer usages to the images that the app creates.
class VulkanStageFrameLoopImageRestore : public VulkanConsumerStage
{
  public:
    VulkanStageFrameLoopImageRestore(CommonObjectInfoTable&    object_info_table,
                                     graphics::FrameLoopInfo&  frame_loop_info,
                                     VulkanDispatchTableLookup dispatch_tables);

    ~VulkanStageFrameLoopImageRestore() override;

    // Loop start

    void StartLooping() override;
    void ProcessStateEndMarker(uint64_t frame_number) override;

    // Devices and images

    void Process_vkDestroyDevice(const ApiCallInfo& call_info, args::DestroyDevice& args) override;
    void Process_vkCreateImage(const ApiCallInfo& call_info, args::CreateImage& args) override;

    // Submits and presents that end a frame

    void Process_vkQueueSubmit(const ApiCallInfo& call_info, args::QueueSubmit& args) override;
    void Process_vkQueueSubmit2(const ApiCallInfo& call_info, args::QueueSubmit2& args) override;
    void Process_vkQueueSubmit2KHR(const ApiCallInfo& call_info, args::QueueSubmit2KHR& args) override;
    void Process_vkQueueBindSparse(const ApiCallInfo& call_info, args::QueueBindSparse& args) override;
    void Process_vkQueuePresentKHR(const ApiCallInfo& call_info, args::QueuePresentKHR& args) override;

  private:
    struct ImageTracking
    {
        ImageTracking(format::HandleId                         device_id,
                      const graphics::VulkanDeviceTable&       device_table,
                      CommonObjectInfoTable&                   object_table,
                      std::shared_ptr<VulkanResourceAllocator> allocator,
                      const VkPhysicalDeviceMemoryProperties*  memory_properties) :
            device_id_(device_id),
            device_table_(device_table), object_table_(object_table), allocator_(allocator),
            memory_properties_(memory_properties)
        {}

        struct ImageState
        {
            graphics::ImageLayoutMap initial_layouts;

            VkImage                               shadow_image{ VK_NULL_HANDLE };
            VkDeviceMemory                        shadow_memory{ VK_NULL_HANDLE };
            VulkanResourceAllocator::ResourceData alloc_data{ 0 };
            VulkanResourceAllocator::MemoryData   mem_data{ 0 };

            std::vector<VkImageSubresourceRange> copyable_ranges;

            std::vector<VkImageSubresourceRange> excluded_ranges;

            std::vector<VkImageCopy> copy_regions;

            VkImageSubresourceRange shadow_range{};

            bool HasShadow() const { return shadow_image != VK_NULL_HANDLE; }
        };

        struct RestoreCommands
        {
            // Puts every copyable subresource of each source image into VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL.
            std::vector<VkImageMemoryBarrier> pre_barriers;

            // Transitions subresources to their initial recorded layouts.
            std::vector<VkImageMemoryBarrier> post_barriers;

            // Images whose contents are copied back out of their shadow.
            std::vector<format::HandleId> copy_ids;

            std::vector<std::pair<format::HandleId, VkImageLayout>> layout_updates;

            bool built{ false };
        };

        void RecordInitialState(const std::vector<format::HandleId>& image_ids,
                                const std::vector<format::HandleId>& restorable_image_ids);

        void Restore(format::HandleId queue);

        void DestroyShadowImages();

        bool CreateShadowImage(format::HandleId image_id, const VulkanImageInfo* image_info, ImageState& state);
        void DestroyShadowImage(ImageState& state);

        void BuildRestoreCommands();

        format::HandleId                                 device_id_;
        const graphics::VulkanDeviceTable&               device_table_;
        CommonObjectInfoTable&                           object_table_;
        std::shared_ptr<VulkanResourceAllocator>         allocator_;
        const VkPhysicalDeviceMemoryProperties*          memory_properties_;
        std::unordered_map<format::HandleId, ImageState> image_states_;
        RestoreCommands                                  restore_commands_;
    };

    ImageTracking& GetImageTracking(format::HandleId device);
    bool           CanSnapshotImageContents(const VulkanImageInfo* image_info) const;
    void           TrackImageStates();
    void           FixupDeviceImages(format::HandleId device, format::HandleId queue);
    void           ResetImageTracking();
    void           ResetImageTracking(format::HandleId device);

    template <typename ArgsT>
    void ProcessQueueSubmit(const ApiCallInfo& call_info, ArgsT& args);

    /// Records the state that each repetition of the loop starts from.
    void RecordLoopStartState();

    /// Waits for the device of the queue, and puts its images back in the state that the loop started from.
    void EndFrame(format::HandleId queue);

    const graphics::VulkanInstanceTable* GetInstanceTable(const void* handle) const;
    const graphics::VulkanDeviceTable*   GetDeviceTable(const void* handle) const;

    CommonObjectInfoTable&    object_info_table_;
    graphics::FrameLoopInfo&  frame_loop_info_;
    VulkanDispatchTableLookup dispatch_tables_;

    /// Images created with the transfer usages, whose contents the stage can copy.
    std::unordered_set<format::HandleId> restorable_images_;

    std::unordered_map<format::HandleId, ImageTracking> per_device_image_tracking_;
};

GFXRECON_END_NAMESPACE(decode)
GFXRECON_END_NAMESPACE(gfxrecon)

#endif // GFXRECON_DECODE_VULKAN_STAGE_FRAME_LOOP_IMAGE_RESTORE_H

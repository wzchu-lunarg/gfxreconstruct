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

// Catch must come before the Vulkan headers.
#include <catch2/catch.hpp>

#include "decode/decode_allocator.h"
#include "decode/vulkan_stage_frame_loop_fixup.h"
#include "generated/generated_vulkan_decoder.h"

#include <cstring>
#include <set>
#include <vector>

using namespace gfxrecon::decode;
namespace format   = gfxrecon::format;
namespace graphics = gfxrecon::graphics;

namespace
{

constexpr format::HandleId kDevice           = 1;
constexpr format::HandleId kQueue            = 10;
constexpr format::HandleId kSignaledFence    = 20;
constexpr format::HandleId kUnsignaledFence  = 21;
constexpr format::HandleId kHostEvent        = 30;
constexpr format::HandleId kDeviceOnlyEvent  = 31;
constexpr format::HandleId kLoopEvent        = 32;
constexpr format::HandleId kSemaphore        = 40;
constexpr format::HandleId kBuffer           = 50;
constexpr uint32_t         kLoopIterations   = 3;
constexpr uint32_t         kSingleRepetition = 2;

// The object info table keeps only infos with a handle. The fake driver below takes the capture IDs back out of them.
// Handles are pointers on 64-bit platforms and integers on 32-bit ones, so copy the bytes.
template <typename HandleT>
HandleT FakeHandle(uint64_t value)
{
    static_assert(sizeof(HandleT) <= sizeof(value));
    HandleT handle{};
    std::memcpy(&handle, &value, sizeof(HandleT));
    return handle;
}

template <typename HandleT>
uint64_t FakeId(HandleT handle)
{
    uint64_t value = 0;
    std::memcpy(&value, &handle, sizeof(HandleT));
    return value;
}

// Stands in for the driver. The dispatch table holds plain function pointers, so they reach the driver of the running
// test through a global.
struct FakeDriver
{
    std::set<uint64_t> signaled_fences;
    std::set<uint64_t> set_events;

    int                   device_waits = 0;
    std::vector<uint64_t> reset_fences;
    std::vector<uint64_t> fence_submits;
    std::vector<uint64_t> event_sets;
    std::vector<uint64_t> event_resets;
    std::vector<uint64_t> waited_semaphores;
    std::vector<uint64_t> signaled_semaphores;
};

FakeDriver* g_driver = nullptr;

VKAPI_ATTR VkResult VKAPI_CALL FakeDeviceWaitIdle(VkDevice)
{
    ++g_driver->device_waits;
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL FakeQueueWaitIdle(VkQueue)
{
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL FakeGetFenceStatus(VkDevice, VkFence fence)
{
    return g_driver->signaled_fences.contains(FakeId(fence)) ? VK_SUCCESS : VK_NOT_READY;
}

VKAPI_ATTR VkResult VKAPI_CALL FakeResetFences(VkDevice, uint32_t count, const VkFence* fences)
{
    for (uint32_t i = 0; i < count; ++i)
    {
        g_driver->reset_fences.push_back(FakeId(fences[i]));
        g_driver->signaled_fences.erase(FakeId(fences[i]));
    }
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL FakeQueueSubmit(VkQueue, uint32_t count, const VkSubmitInfo* submits, VkFence fence)
{
    for (uint32_t i = 0; i < count; ++i)
    {
        for (uint32_t j = 0; j < submits[i].waitSemaphoreCount; ++j)
        {
            g_driver->waited_semaphores.push_back(FakeId(submits[i].pWaitSemaphores[j]));
        }
        for (uint32_t j = 0; j < submits[i].signalSemaphoreCount; ++j)
        {
            g_driver->signaled_semaphores.push_back(FakeId(submits[i].pSignalSemaphores[j]));
        }
    }

    if (fence != VK_NULL_HANDLE)
    {
        g_driver->fence_submits.push_back(FakeId(fence));
        g_driver->signaled_fences.insert(FakeId(fence));
    }
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL FakeGetEventStatus(VkDevice, VkEvent event)
{
    return g_driver->set_events.contains(FakeId(event)) ? VK_EVENT_SET : VK_EVENT_RESET;
}

VKAPI_ATTR VkResult VKAPI_CALL FakeSetEvent(VkDevice, VkEvent event)
{
    g_driver->event_sets.push_back(FakeId(event));
    g_driver->set_events.insert(FakeId(event));
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL FakeResetEvent(VkDevice, VkEvent event)
{
    g_driver->event_resets.push_back(FakeId(event));
    g_driver->set_events.erase(FakeId(event));
    return VK_SUCCESS;
}

std::vector<uint8_t> CreateBufferBytes(format::HandleId buffer, VkBufferUsageFlags usage)
{
    return ByteWriter()
        .Put<format::HandleId>(kDevice)                      // device
        .Put<uint32_t>(kStructPointer)                       // pCreateInfo
        .Put<uint64_t>(kUnusedAddress)                       //   address
        .Put<uint32_t>(VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO) //   sType
        .Put<uint32_t>(kNullStructPointer)                   //   pNext
        .Put<uint32_t>(0)                                    //   flags
        .Put<uint64_t>(256)                                  //   size
        .Put<uint32_t>(usage)                                //   usage
        .Put<uint32_t>(VK_SHARING_MODE_EXCLUSIVE)            //   sharingMode
        .Put<uint32_t>(0)                                    //   queueFamilyIndexCount
        .Put<uint32_t>(kNullArray)                           //   pQueueFamilyIndices
        .Put<uint32_t>(kNullStructPointer)                   // pAllocator
        .Put<uint32_t>(kHandlePointer)                       // pBuffer
        .Put<uint64_t>(kUnusedAddress)                       //   address
        .Put<format::HandleId>(buffer)                       //   id
        .Put<int32_t>(VK_SUCCESS)                            // result
        .Bytes();
}

// vkCreateFence, vkCreateEvent and vkCreateSemaphore take a create info with only flags.
std::vector<uint8_t> CreateObjectBytes(VkStructureType s_type, uint32_t flags, format::HandleId object)
{
    return ByteWriter()
        .Put<format::HandleId>(kDevice)    // device
        .Put<uint32_t>(kStructPointer)     // pCreateInfo
        .Put<uint64_t>(kUnusedAddress)     //   address
        .Put<uint32_t>(s_type)             //   sType
        .Put<uint32_t>(kNullStructPointer) //   pNext
        .Put<uint32_t>(flags)              //   flags
        .Put<uint32_t>(kNullStructPointer) // pAllocator
        .Put<uint32_t>(kHandlePointer)     // pObject
        .Put<uint64_t>(kUnusedAddress)     //   address
        .Put<format::HandleId>(object)     //   id
        .Put<int32_t>(VK_SUCCESS)          // result
        .Bytes();
}

std::vector<uint8_t> DestroyEventBytes(format::HandleId event)
{
    return ByteWriter()
        .Put<format::HandleId>(kDevice)    // device
        .Put<format::HandleId>(event)      // event
        .Put<uint32_t>(kNullStructPointer) // pAllocator
        .Bytes();
}

// One submit info that signals one semaphore, with no command buffers and no fence.
std::vector<uint8_t> QueueSubmitSignalBytes(format::HandleId semaphore)
{
    return ByteWriter()
        .Put<format::HandleId>(kQueue)                // queue
        .Put<uint32_t>(1)                             // submitCount
        .Put<uint32_t>(kStructArrayPointer)           // pSubmits
        .Put<uint64_t>(kUnusedAddress)                //   address
        .Put<uint64_t>(1)                             //   length
        .Put<uint32_t>(VK_STRUCTURE_TYPE_SUBMIT_INFO) //   sType
        .Put<uint32_t>(kNullStructPointer)            //   pNext
        .Put<uint32_t>(0)                             //   waitSemaphoreCount
        .Put<uint32_t>(kNullArray)                    //   pWaitSemaphores
        .Put<uint32_t>(kNullArray)                    //   pWaitDstStageMask
        .Put<uint32_t>(0)                             //   commandBufferCount
        .Put<uint32_t>(kNullArray)                    //   pCommandBuffers
        .Put<uint32_t>(1)                             //   signalSemaphoreCount
        .Put<uint32_t>(kArrayPointer)                 //   pSignalSemaphores
        .Put<uint64_t>(kUnusedAddress)                //     address
        .Put<uint64_t>(1)                             //     length
        .Put<format::HandleId>(semaphore)             //     ids
        .Put<format::HandleId>(format::kNullHandleId) // fence
        .Put<int32_t>(VK_SUCCESS)                     // result
        .Bytes();
}

// A present without semaphores or swapchains. The stage only needs the queue.
std::vector<uint8_t> QueuePresentBytes()
{
    return ByteWriter()
        .Put<format::HandleId>(kQueue)                     // queue
        .Put<uint32_t>(kStructPointer)                     // pPresentInfo
        .Put<uint64_t>(kUnusedAddress)                     //   address
        .Put<uint32_t>(VK_STRUCTURE_TYPE_PRESENT_INFO_KHR) //   sType
        .Put<uint32_t>(kNullStructPointer)                 //   pNext
        .Put<uint32_t>(0)                                  //   waitSemaphoreCount
        .Put<uint32_t>(kNullArray)                         //   pWaitSemaphores
        .Put<uint32_t>(0)                                  //   swapchainCount
        .Put<uint32_t>(kNullArray)                         //   pSwapchains
        .Put<uint32_t>(kNullArray)                         //   pImageIndices
        .Put<uint32_t>(kNullArray)                         //   pResults
        .Put<int32_t>(VK_SUCCESS)                          // result
        .Bytes();
}

// Stands in for the replay consumer, which creates the objects and fills the object info table.
class FakeReplayConsumer : public VulkanConsumer
{
  public:
    FakeReplayConsumer(CommonObjectInfoTable& table, FakeDriver& driver) : table_(table), driver_(driver) {}

    void Process_vkCreateBuffer(const ApiCallInfo&, args::CreateBuffer& args) override
    {
        buffer_usage = args.pCreateInfo.GetPointer()->usage;
    }

    void Process_vkCreateFence(const ApiCallInfo&, args::CreateFence& args) override
    {
        VulkanFenceInfo info;
        info.capture_id = *args.pFence.GetPointer();
        info.handle     = FakeHandle<VkFence>(info.capture_id);
        info.parent_id  = args.device;
        table_.AddVkFenceInfo(std::move(info));

        if ((args.pCreateInfo.GetPointer()->flags & VK_FENCE_CREATE_SIGNALED_BIT) != 0)
        {
            driver_.signaled_fences.insert(*args.pFence.GetPointer());
        }
    }

    void Process_vkCreateEvent(const ApiCallInfo&, args::CreateEvent& args) override
    {
        VulkanEventInfo info;
        info.capture_id = *args.pEvent.GetPointer();
        info.handle     = FakeHandle<VkEvent>(info.capture_id);
        info.parent_id  = args.device;
        table_.AddVkEventInfo(std::move(info));
    }

    void Process_vkDestroyEvent(const ApiCallInfo&, args::DestroyEvent& args) override
    {
        table_.RemoveVkEventInfo(args.event);
    }

    void Process_vkCreateSemaphore(const ApiCallInfo&, args::CreateSemaphore& args) override
    {
        VulkanSemaphoreInfo info;
        info.capture_id = *args.pSemaphore.GetPointer();
        info.handle     = FakeHandle<VkSemaphore>(info.capture_id);
        info.parent_id  = args.device;
        table_.AddVkSemaphoreInfo(std::move(info));
    }

    void Process_vkQueueSubmit(const ApiCallInfo&, args::QueueSubmit&) override { ++submits; }
    void Process_vkQueuePresentKHR(const ApiCallInfo&, args::QueuePresentKHR&) override { ++presents; }

    void WaitDevicesIdle() override { ++device_idle_waits; }

    VkBufferUsageFlags buffer_usage      = 0;
    int                submits           = 0;
    int                presents          = 0;
    int                device_idle_waits = 0;

  private:
    CommonObjectInfoTable& table_;
    FakeDriver&            driver_;
};

class Harness
{
  public:
    explicit Harness(uint32_t loop_iterations = kLoopIterations) :
        frame_loop_info(1, loop_iterations), consumer(table, driver),
        stage(table, frame_loop_info, MakeDispatchTables(instance_table_, device_table_))
    {
        g_driver = &driver;

        device_table_.DeviceWaitIdle = FakeDeviceWaitIdle;
        device_table_.QueueWaitIdle  = FakeQueueWaitIdle;
        device_table_.QueueSubmit    = FakeQueueSubmit;
        device_table_.GetFenceStatus = FakeGetFenceStatus;
        device_table_.ResetFences    = FakeResetFences;
        device_table_.GetEventStatus = FakeGetEventStatus;
        device_table_.SetEvent       = FakeSetEvent;
        device_table_.ResetEvent     = FakeResetEvent;

        decoder_.AddConsumer(&stage);
        stage.AddNext(&consumer);

        VulkanDeviceInfo device_info;
        device_info.capture_id = kDevice;
        device_info.handle     = FakeHandle<VkDevice>(kDevice);
        table.AddVkDeviceInfo(std::move(device_info));

        VulkanQueueInfo queue_info;
        queue_info.capture_id   = kQueue;
        queue_info.handle       = FakeHandle<VkQueue>(kQueue);
        queue_info.parent       = FakeHandle<VkDevice>(kDevice);
        queue_info.parent_id    = kDevice;
        queue_info.family_index = 0;
        queue_info.queue_index  = 0;
        table.AddVkQueueInfo(std::move(queue_info));
    }

    ~Harness() { g_driver = nullptr; }

    void Send(format::ApiCallId call_id, const std::vector<uint8_t>& bytes)
    {
        DecoderAllocGuard<true> guard;
        decoder_.DecodeFunctionCall(call_id, ApiCallInfo{}, bytes.data(), bytes.size());
    }

    // The application starts the loop at the loop frame, and the preloaded file processor tells the decoder.
    void StartLoop()
    {
        frame_loop_info.SetLooping(true);
        decoder_.StartLooping();
    }

    // The application moves on to the next iteration once a frame ends.
    void NextIteration() { frame_loop_info.DecrementLoopIterations(); }

    // The table and the driver come first, because the stage and the consumer keep references to them.
    CommonObjectInfoTable   table;
    FakeDriver              driver;
    graphics::FrameLoopInfo frame_loop_info;
    FakeReplayConsumer      consumer;

  private:
    static VulkanDispatchTableLookup MakeDispatchTables(const graphics::VulkanInstanceTable& instance_table,
                                                        const graphics::VulkanDeviceTable&   device_table)
    {
        VulkanDispatchTableLookup lookup;
        lookup.get_instance_table = [&instance_table](const void*) { return &instance_table; };
        lookup.get_device_table   = [&device_table](const void*) { return &device_table; };
        return lookup;
    }

    graphics::VulkanInstanceTable instance_table_;
    graphics::VulkanDeviceTable   device_table_;
    VulkanDecoder                 decoder_;

  public:
    VulkanStageFrameLoopFixup stage;
};

} // namespace

TEST_CASE("Frame loop fixup stage lets buffers be copied", "[stage][frame_loop_fixup]")
{
    Harness harness;
    harness.Send(format::ApiCall_vkCreateBuffer, CreateBufferBytes(kBuffer, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT));

    CHECK(harness.consumer.buffer_usage ==
          (VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT));
}

TEST_CASE("Frame loop fixup stage does nothing before the loop starts", "[stage][frame_loop_fixup]")
{
    Harness harness;
    harness.Send(format::ApiCall_vkCreateFence,
                 CreateObjectBytes(VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, VK_FENCE_CREATE_SIGNALED_BIT, kSignaledFence));
    harness.Send(format::ApiCall_vkQueuePresentKHR, QueuePresentBytes());

    CHECK(harness.consumer.presents == 1);
    CHECK(harness.driver.device_waits == 0);
    CHECK(harness.driver.reset_fences.empty());
}

TEST_CASE("Frame loop fixup stage puts fences back in their state at the loop start", "[stage][frame_loop_fixup]")
{
    Harness harness;
    harness.Send(format::ApiCall_vkCreateFence,
                 CreateObjectBytes(VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, VK_FENCE_CREATE_SIGNALED_BIT, kSignaledFence));
    harness.Send(format::ApiCall_vkCreateFence,
                 CreateObjectBytes(VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, 0, kUnsignaledFence));

    harness.StartLoop();

    // The stage waits for the devices before it reads their state.
    CHECK(harness.consumer.device_idle_waits == 1);

    // The frame swaps the states of the fences.
    harness.driver.signaled_fences = { kUnsignaledFence };

    harness.Send(format::ApiCall_vkQueuePresentKHR, QueuePresentBytes());

    CHECK(harness.consumer.presents == 1);
    CHECK(harness.driver.device_waits == 1);
    CHECK(std::set<uint64_t>(harness.driver.reset_fences.begin(), harness.driver.reset_fences.end()) ==
          std::set<uint64_t>{ kSignaledFence, kUnsignaledFence });
    CHECK(harness.driver.fence_submits == std::vector<uint64_t>{ kSignaledFence });
    CHECK(harness.driver.signaled_fences == std::set<uint64_t>{ kSignaledFence });
}

TEST_CASE("Frame loop fixup stage leaves the state alone after the final iteration", "[stage][frame_loop_fixup]")
{
    Harness harness(kSingleRepetition);
    harness.Send(format::ApiCall_vkCreateFence,
                 CreateObjectBytes(VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, VK_FENCE_CREATE_SIGNALED_BIT, kSignaledFence));

    harness.StartLoop();
    harness.Send(format::ApiCall_vkQueuePresentKHR, QueuePresentBytes());
    CHECK(harness.driver.fence_submits.size() == 1);

    harness.NextIteration();
    REQUIRE(harness.frame_loop_info.IsFinalIteration());
    harness.Send(format::ApiCall_vkQueuePresentKHR, QueuePresentBytes());

    // The stage still waits for the frame, but nothing repeats after it.
    CHECK(harness.driver.device_waits == 2);
    CHECK(harness.driver.fence_submits.size() == 1);
}

TEST_CASE("Frame loop fixup stage puts host visible events back in their state at the loop start",
          "[stage][frame_loop_fixup]")
{
    Harness harness;
    harness.Send(format::ApiCall_vkCreateEvent, CreateObjectBytes(VK_STRUCTURE_TYPE_EVENT_CREATE_INFO, 0, kHostEvent));
    harness.Send(
        format::ApiCall_vkCreateEvent,
        CreateObjectBytes(VK_STRUCTURE_TYPE_EVENT_CREATE_INFO, VK_EVENT_CREATE_DEVICE_ONLY_BIT, kDeviceOnlyEvent));
    harness.driver.set_events = { kHostEvent, kDeviceOnlyEvent };

    harness.StartLoop();
    harness.driver.set_events.clear();
    harness.Send(format::ApiCall_vkQueuePresentKHR, QueuePresentBytes());

    // The host cannot set a device only event, so the stage skips it.
    CHECK(harness.driver.event_sets == std::vector<uint64_t>{ kHostEvent });
    CHECK(harness.driver.event_resets.empty());
    CHECK(harness.table.GetVkEventInfo(kHostEvent)->latched_set);
}

TEST_CASE("Frame loop fixup stage forgets an event that the loop creates and destroys", "[stage][frame_loop_fixup]")
{
    Harness harness;
    harness.StartLoop();

    harness.Send(format::ApiCall_vkCreateEvent, CreateObjectBytes(VK_STRUCTURE_TYPE_EVENT_CREATE_INFO, 0, kLoopEvent));
    harness.Send(format::ApiCall_vkDestroyEvent, DestroyEventBytes(kLoopEvent));
    harness.Send(format::ApiCall_vkQueuePresentKHR, QueuePresentBytes());

    CHECK(harness.driver.event_sets.empty());
    CHECK(harness.driver.event_resets.empty());
}

TEST_CASE("Frame loop fixup stage waits on a semaphore that the loop leaves signaled", "[stage][frame_loop_fixup]")
{
    Harness harness;
    harness.Send(format::ApiCall_vkCreateSemaphore,
                 CreateObjectBytes(VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, 0, kSemaphore));

    harness.StartLoop();
    harness.Send(format::ApiCall_vkQueueSubmit, QueueSubmitSignalBytes(kSemaphore));
    CHECK(harness.consumer.submits == 1);

    // A submit that ends no frame leaves the semaphore to the present.
    CHECK(harness.driver.waited_semaphores.empty());

    harness.Send(format::ApiCall_vkQueuePresentKHR, QueuePresentBytes());

    CHECK(harness.driver.waited_semaphores == std::vector<uint64_t>{ kSemaphore });
    CHECK(harness.driver.signaled_semaphores.empty());
}

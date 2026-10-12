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

/*
** A replay event plugin that polls Arm Mali hardware counters with libGPUCounters at a fixed period
** and appends both the counter readings and every replay event it receives to one text file.
**
** --replay-event-plugin-params takes the output path, optionally followed by the polling period:
**
**     /path/to/counters.txt period_us=1000
**
** The period defaults to kDefaultPeriodUs. Each line is whitespace separated key=value pairs and
** starts with kind=sample or kind=event:
**
**     kind=sample timestamp_ns=94518200 interval_ns=1000412 MaliGPUActiveCy=812000 ...
**     kind=event type=4 abi=4 size=48 frame=1 timestamp_ns=94518731 first_submit_index=3 ...
*/

#include <gfxr/replay_event_plugin.h>

#include <hwcpipe/gpu.hpp>
#include <hwcpipe/hwcpipe_counter.h>
#include <hwcpipe/sampler.hpp>

#include <util/date_time.h>
#include <util/logging.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdio.h>
#include <string>
#include <system_error>
#include <thread>
#include <vector>

// The ABI version that carries WarmupBegin and WarmupEnd
static const uint32_t kRequiredAbiVersion = 4u;

static const int kDeviceNumber = 0;

static const uint32_t kDefaultPeriodUs = 1000;
static const uint32_t kMinPeriodUs     = 100;
static const uint32_t kMaxPeriodUs     = 1000000;

typedef struct CounterEntry
{
    // The identifier as libGPUCounters spells it in its hwcpipe_counter enum
    const char*     name;
    hwcpipe_counter counter;
} CounterEntry;

// The counters to sample.
static const CounterEntry kCounters[] = {
    { "MaliGPUActiveCy", MaliGPUActiveCy },
    { "MaliCompQueueActiveCy", MaliCompQueueActiveCy },
    { "MaliVertQueueActiveCy", MaliVertQueueActiveCy },
    { "MaliNonFragQueueActiveCy", MaliNonFragQueueActiveCy },
    { "MaliFragQueueActiveCy", MaliFragQueueActiveCy },
    { "MaliClockRatioSC", MaliClockRatioSC },
};

typedef struct CounterRecorderPlugin
{
    // Must stay first, because replay casts the plugin pointer to this base
    GfxrReplayPluginV1 base   = {};
    FILE*              output = nullptr;

    // Serializes writes to output between the sampler thread and replay's thread
    std::mutex output_mutex;

    std::unique_ptr<hwcpipe::sampler<>> sampler;
    bool                                sampling = false;

    // The entries of kCounters this GPU accepted, in output order
    std::vector<const CounterEntry*> counters;

    std::chrono::microseconds period{ kDefaultPeriodUs };

    // Sampler thread state. stop is guarded by stop_mutex.
    std::thread             sampler_thread;
    std::mutex              stop_mutex;
    std::condition_variable stop_condition;
    bool                    stop = false;

    // Only touched by the sampler thread
    int64_t last_sample_timestamp_ns = 0;
} CounterRecorderPlugin;

static void stop_sampler_thread(CounterRecorderPlugin* recorder)
{
    if (!recorder->sampler_thread.joinable())
    {
        return;
    }

    {
        std::lock_guard<std::mutex> lock(recorder->stop_mutex);
        recorder->stop = true;
    }
    recorder->stop_condition.notify_all();
    recorder->sampler_thread.join();
}

static void destroy(GfxrReplayPluginV1* self)
{
    CounterRecorderPlugin* recorder = reinterpret_cast<CounterRecorderPlugin*>(self);
    if (recorder == nullptr)
    {
        return;
    }

    // The thread owns the sampler while it runs, so it has to be gone before sampling stops.
    stop_sampler_thread(recorder);

    if (recorder->sampling)
    {
        const std::error_code stop_error = recorder->sampler->stop_sampling();
        if (stop_error)
        {
            GFXRECON_LOG_WARNING("Replay event counter recorder: failed to stop sampling (%s)",
                                 stop_error.message().c_str());
        }
    }

    if (recorder->output != nullptr)
    {
        fclose(recorder->output);
    }

    delete recorder;

    gfxrecon::util::Log::Release();
}

static void append_unsigned(FILE* output, const char* key, uint64_t value)
{
    fprintf(output, " %s=%llu", key, (unsigned long long)value);
}

static void append_signed(FILE* output, const char* key, int64_t value)
{
    fprintf(output, " %s=%lld", key, (long long)value);
}

static void append_counter(FILE* output, const char* key, double value)
{
    // Cycle counters are whole numbers and read better without a fraction. A ratio needs one.
    if (value == std::floor(value))
    {
        fprintf(output, " %s=%.0f", key, value);
    }
    else
    {
        fprintf(output, " %s=%.3f", key, value);
    }
}

// Reads the counters accumulated since the previous reading and writes one sample line. Runs on
// the sampler thread only.
static void write_sample(CounterRecorderPlugin* recorder, std::vector<double>& values)
{
    // A refused reading leaves the counters accumulating, so the interval keeps growing until a
    // reading succeeds.
    bool sample_ok = !recorder->sampler->sample_now();

    const int64_t timestamp_ns = gfxrecon::util::datetime::GetTimestamp();

    for (size_t index = 0; sample_ok && (index < recorder->counters.size()); ++index)
    {
        hwcpipe::counter_sample sample;

        if (recorder->sampler->get_counter_value(recorder->counters[index]->counter, sample))
        {
            sample_ok = false;
            break;
        }

        values[index] = (sample.type == hwcpipe::counter_sample::type::float64)
                            ? sample.value.float64
                            : static_cast<double>(sample.value.uint64);
    }

    std::lock_guard<std::mutex> lock(recorder->output_mutex);

    fprintf(recorder->output, "kind=sample");
    append_unsigned(recorder->output, "timestamp_ns", static_cast<uint64_t>(timestamp_ns));
    append_unsigned(
        recorder->output, "interval_ns", static_cast<uint64_t>(timestamp_ns - recorder->last_sample_timestamp_ns));

    if (sample_ok)
    {
        for (size_t index = 0; index < recorder->counters.size(); ++index)
        {
            append_counter(recorder->output, recorder->counters[index]->name, values[index]);
        }

        recorder->last_sample_timestamp_ns = timestamp_ns;
    }
    else
    {
        append_unsigned(recorder->output, "sample_error", 1);
    }

    fputc('\n', recorder->output);

    // Android replay is terminated without destroying the plugin, so every line has to reach the file
    // as soon as it is written.
    fflush(recorder->output);
}

static void sampler_thread_main(CounterRecorderPlugin* recorder)
{
    std::vector<double> values(recorder->counters.size(), 0.0);

    auto next = std::chrono::steady_clock::now() + recorder->period;

    std::unique_lock<std::mutex> lock(recorder->stop_mutex);
    while (!recorder->stop_condition.wait_until(lock, next, [recorder] { return recorder->stop; }))
    {
        lock.unlock();
        write_sample(recorder, values);
        lock.lock();

        // Keep a steady cadence, but do not try to catch up on periods a slow reading overran.
        next += recorder->period;
        const auto now = std::chrono::steady_clock::now();
        if (next < now)
        {
            next = now + recorder->period;
        }
    }
}

static bool payload_fits(FILE* output, const GfxrReplayEventHeader* event, size_t expected_size)
{
    if (event->struct_size >= expected_size)
    {
        return true;
    }

    append_unsigned(output, "truncated", 1);
    return false;
}

// Returns false when the event was too small for its type's struct.
static bool append_payload(FILE* output, const GfxrReplayEventHeader* event)
{
    switch (event->type)
    {
        case GFXR_REPLAY_EVENT_QUEUE_SUBMIT_BEGIN:
        {
            if (!payload_fits(output, event, sizeof(GfxrReplayQueueSubmitBeginEvent)))
            {
                return false;
            }

            const GfxrReplayQueueSubmitBeginEvent* submit_begin =
                reinterpret_cast<const GfxrReplayQueueSubmitBeginEvent*>(event);

            append_unsigned(output, "submit_index", submit_begin->submit_index);
            append_unsigned(output, "queue_id", submit_begin->queue_id);
            break;
        }

        case GFXR_REPLAY_EVENT_QUEUE_SUBMIT_END:
        {
            if (!payload_fits(output, event, sizeof(GfxrReplayQueueSubmitEndEvent)))
            {
                return false;
            }

            const GfxrReplayQueueSubmitEndEvent* submit_end =
                reinterpret_cast<const GfxrReplayQueueSubmitEndEvent*>(event);

            append_unsigned(output, "submit_index", submit_end->submit_index);
            append_unsigned(output, "queue_id", submit_end->queue_id);
            // VkResult, is negative for a Vulkan error
            append_signed(output, "result", submit_end->result);
            append_unsigned(output, "completion_source", submit_end->completion_source);
            break;
        }

        case GFXR_REPLAY_EVENT_FRAME_END:
        {
            if (!payload_fits(output, event, sizeof(GfxrReplayFrameEndEvent)))
            {
                return false;
            }

            const GfxrReplayFrameEndEvent* frame_end = reinterpret_cast<const GfxrReplayFrameEndEvent*>(event);

            append_unsigned(output, "first_submit_index", frame_end->first_submit_index);
            append_unsigned(output, "last_submit_index", frame_end->last_submit_index);
            break;
        }

        case GFXR_REPLAY_EVENT_WAIT_BEGIN:
        {
            if (!payload_fits(output, event, sizeof(GfxrReplayWaitBeginEvent)))
            {
                return false;
            }

            const GfxrReplayWaitBeginEvent* wait_begin = reinterpret_cast<const GfxrReplayWaitBeginEvent*>(event);

            append_unsigned(output, "requested_duration_ms", wait_begin->requested_duration_ms);
            break;
        }

        case GFXR_REPLAY_EVENT_WARMUP_BEGIN:
        {
            if (!payload_fits(output, event, sizeof(GfxrReplayWarmupBeginEvent)))
            {
                return false;
            }

            const GfxrReplayWarmupBeginEvent* warmup_begin = reinterpret_cast<const GfxrReplayWarmupBeginEvent*>(event);

            append_unsigned(output, "warm_up_load", warmup_begin->warm_up_load);
            break;
        }

        // These carry nothing beyond the header
        case GFXR_REPLAY_EVENT_FRAME_BEGIN:
        case GFXR_REPLAY_EVENT_STATE_SETUP_BEGIN:
        case GFXR_REPLAY_EVENT_STATE_SETUP_END:
        case GFXR_REPLAY_EVENT_WAIT_END:
        case GFXR_REPLAY_EVENT_WARMUP_END:
            break;

        // Unknown event types such as from a newer ABI
        default:
            append_unsigned(output, "unknown_type", 1);
            break;
    }

    return true;
}

static GfxrReplayPluginResult on_event(GfxrReplayPluginV1* self, const GfxrReplayEventHeader* event)
{
    CounterRecorderPlugin* recorder = reinterpret_cast<CounterRecorderPlugin*>(self);
    if ((recorder == nullptr) || (recorder->output == nullptr) || (event == nullptr))
    {
        return GFXR_REPLAY_PLUGIN_RESULT_ERROR;
    }

    std::lock_guard<std::mutex> lock(recorder->output_mutex);

    fprintf(recorder->output, "kind=event type=%u", (unsigned int)event->type);
    append_unsigned(recorder->output, "abi", event->abi_version);
    append_unsigned(recorder->output, "size", event->struct_size);
    append_unsigned(recorder->output, "frame", event->frame_index);
    append_unsigned(recorder->output, "timestamp_ns", event->timestamp_ns);

    bool payload_complete = append_payload(recorder->output, event);

    fputc('\n', recorder->output);

    fflush(recorder->output);

    return payload_complete ? GFXR_REPLAY_PLUGIN_RESULT_OK : GFXR_REPLAY_PLUGIN_RESULT_ERROR;
}

// Splits --replay-event-plugin-params into the output path and the optional period_us=N. Returns
// false and logs the reason when the parameters are malformed.
static bool parse_params(const char* params, std::string& output_path, std::chrono::microseconds& period)
{
    std::istringstream tokens((params != nullptr) ? params : "");
    std::string        token;

    while (tokens >> token)
    {
        static const std::string kPeriodKey = "period_us=";

        if (token.compare(0, kPeriodKey.size(), kPeriodKey) == 0)
        {
            const std::string   value     = token.substr(kPeriodKey.size());
            char*               end       = nullptr;
            const unsigned long period_us = strtoul(value.c_str(), &end, 10);

            if (value.empty() || (*end != '\0') || (period_us < kMinPeriodUs) || (period_us > kMaxPeriodUs))
            {
                GFXRECON_LOG_ERROR("Replay event counter recorder: '%s' is not a period between %u and %u "
                                   "microseconds",
                                   token.c_str(),
                                   kMinPeriodUs,
                                   kMaxPeriodUs);
                return false;
            }

            period = std::chrono::microseconds(period_us);
        }
        else if (output_path.empty())
        {
            output_path = token;
        }
        else
        {
            GFXRECON_LOG_ERROR("Replay event counter recorder: unexpected parameter '%s'", token.c_str());
            return false;
        }
    }

    if (output_path.empty())
    {
        GFXRECON_LOG_ERROR("Replay event counter recorder: --replay-event-plugin-params must name an output file");
        return false;
    }

    return true;
}

static bool start_sampling(CounterRecorderPlugin* recorder)
{
    hwcpipe::gpu gpu(kDeviceNumber);
    if (!gpu)
    {
        GFXRECON_LOG_ERROR("Replay event counter recorder: Mali GPU device %d was not found. The plugin needs an "
                           "Arm Mali GPU and read access to its kernel counter interface.",
                           kDeviceNumber);
        return false;
    }

    // The sampler copies what it needs out of the config, so this does not have to outlive it.
    hwcpipe::sampler_config config(gpu);

    for (const CounterEntry& entry : kCounters)
    {
        const std::error_code add_error = config.add_counter(entry.counter);
        if (add_error)
        {
            GFXRECON_LOG_WARNING("Replay event counter recorder: %s is not available on this GPU (%s)",
                                 entry.name,
                                 add_error.message().c_str());
            continue;
        }
        recorder->counters.push_back(&entry);
    }

    if (recorder->counters.empty())
    {
        GFXRECON_LOG_ERROR("Replay event counter recorder: none of the sampled counters are available on this GPU");
        return false;
    }

    recorder->sampler = std::make_unique<hwcpipe::sampler<>>(config);
    if (!static_cast<bool>(*recorder->sampler))
    {
        GFXRECON_LOG_ERROR("Replay event counter recorder: libGPUCounters could not create a sampler for this GPU");
        return false;
    }

    const std::error_code start_error = recorder->sampler->start_sampling();
    if (start_error)
    {
        GFXRECON_LOG_ERROR("Replay event counter recorder: libGPUCounters could not start sampling: %s",
                           start_error.message().c_str());
        return false;
    }

    recorder->sampling                 = true;
    recorder->last_sample_timestamp_ns = gfxrecon::util::datetime::GetTimestamp();

    return true;
}

GFXR_REPLAY_PLUGIN_EXPORT GfxrReplayPluginV1* gfxrCreateReplayPluginV1(const GfxrReplayPluginCreateInfo* create_info)
{
    gfxrecon::util::Log::Init();

    if ((create_info == nullptr) || (create_info->struct_size != sizeof(GfxrReplayPluginCreateInfo)))
    {
        GFXRECON_LOG_ERROR("Replay event counter recorder: invalid create info");
        gfxrecon::util::Log::Release();
        return nullptr;
    }

    const uint32_t negotiated_version = std::min(create_info->abi_version, GFXR_REPLAY_PLUGIN_ABI_VERSION);
    if (negotiated_version < kRequiredAbiVersion)
    {
        GFXRECON_LOG_ERROR("Replay event counter recorder: replay offers plugin ABI version %u, but this plugin needs "
                           "%u for the frame warm-up events",
                           create_info->abi_version,
                           kRequiredAbiVersion);
        gfxrecon::util::Log::Release();
        return nullptr;
    }

    std::string               output_path;
    std::chrono::microseconds period(kDefaultPeriodUs);
    if (!parse_params(create_info->plugin_params, output_path, period))
    {
        gfxrecon::util::Log::Release();
        return nullptr;
    }

    CounterRecorderPlugin* recorder = new CounterRecorderPlugin();

    recorder->base.abi_version = negotiated_version;
    recorder->base.struct_size = sizeof(GfxrReplayPluginV1);
    recorder->base.destroy     = destroy;
    recorder->base.on_event    = on_event;
    recorder->period           = period;

    if (!start_sampling(recorder))
    {
        destroy(&recorder->base);
        return nullptr;
    }

    recorder->output = fopen(output_path.c_str(), "w");
    if (recorder->output == nullptr)
    {
        GFXRECON_LOG_ERROR("Replay event counter recorder: failed to open '%s' for writing", output_path.c_str());
        destroy(&recorder->base);
        return nullptr;
    }

    // From here on the sampler belongs to the thread until destroy() joins it.
    recorder->sampler_thread = std::thread(sampler_thread_main, recorder);

    GFXRECON_LOG_INFO("Replay event counter recorder: polling %zu counters every %lld us into %s",
                      recorder->counters.size(),
                      static_cast<long long>(recorder->period.count()),
                      output_path.c_str());

    return &recorder->base;
}

const std = @import("std");
const builtin = @import("builtin");
const c = @import("c.zig").c;
const FixedArray = @import("fixed_array.zig").FixedArray;
const platform = @import("platform.zig");
const vk_shared = @import("vk_shared.zig");
const vk_check = vk_shared.vk_check;
const logger = @import("logger.zig");

pub const VulkanInstance = struct {
    handle: c.VkInstance,
    debug_messenger: c.VkDebugUtilsMessengerEXT,
    const Self = @This();

    pub fn init(self: *VulkanInstance) bool {
        var req_extensions: FixedArray([*c]const u8, 3) = .init_from_slice(&[_][*c]const u8{
            c.VK_KHR_SURFACE_EXTENSION_NAME,
            c.VK_EXT_DEBUG_UTILS_EXTENSION_NAME,
        });
        req_extensions.append(platform.get_vk_platform_extension());

        var avail_extension_cnt: u32 = undefined;
        vk_check(c.vkEnumerateInstanceExtensionProperties.?(null, &avail_extension_cnt, null));

        const MAX_EXTENSIONS: u32 = 40;
        var avail_extensions: FixedArray(c.VkExtensionProperties, MAX_EXTENSIONS) = .init_len(avail_extension_cnt);
        vk_check(c.vkEnumerateInstanceExtensionProperties.?(null, &avail_extension_cnt, avail_extensions.data()));

        outer: for (req_extensions.slice_full()) |req_ext| {
            for (avail_extensions.slice_full()) |avail_ext| {
                const req_ext_slice = std.mem.span(req_ext);
                if (std.mem.eql(u8, req_ext_slice, avail_ext.extensionName[0..req_ext_slice.len])) {
                    continue :outer;
                }
            }
            std.log.debug("extension {s} was not found", .{req_ext});
            return false;
        }

        const app_info = std.mem.zeroInit(c.VkApplicationInfo, .{
            .sType = c.VK_STRUCTURE_TYPE_APPLICATION_INFO,
            .apiVersion = c.VK_API_VERSION_1_4,
        });

        var instance_info = std.mem.zeroInit(c.VkInstanceCreateInfo, .{
            .sType = c.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            .pApplicationInfo = &app_info,
            .enabledExtensionCount = req_extensions.len,
            .ppEnabledExtensionNames = req_extensions.c_ptr(),
        });

        if (builtin.mode == .Debug) {
            var req_layers: FixedArray([*c]const u8, 1) = .init_from_slice(&[_][*c]const u8{"VK_LAYER_KHRONOS_validation"});
            var avail_layers_cnt: u32 = undefined;
            vk_check(c.vkEnumerateInstanceLayerProperties.?(&avail_layers_cnt, null));

            const MAX_LAYERS: u32 = 30;
            var avail_layers: FixedArray(c.VkLayerProperties, MAX_LAYERS) = .init_len(avail_layers_cnt);
            vk_check(c.vkEnumerateInstanceLayerProperties.?(&avail_layers_cnt, avail_layers.data()));

            outer: for (req_layers.slice_full()) |req_layer| {
                for (avail_layers.slice_full()) |avail_layer| {
                    const req_layer_slice = std.mem.span(req_layer);
                    if (std.mem.eql(u8, req_layer_slice, avail_layer.layerName[0..req_layer_slice.len])) {
                        continue :outer;
                    }
                }
                std.log.debug("Required layer was not found: {s}", .{std.mem.span(req_layer)});
                return false;
            }

            instance_info.enabledLayerCount = req_layers.len;
            instance_info.ppEnabledLayerNames = req_layers.data();
        }

        vk_check(c.vkCreateInstance.?(&instance_info, null, &self.handle));
        _ = c.volkLoadInstance(self.handle);

        if (builtin.mode == .Debug) {
            const log_severity: u32 = c.VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT |
                c.VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                c.VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT;

            const create_info: c.VkDebugUtilsMessengerCreateInfoEXT = .{
                .sType = c.VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
                .messageSeverity = log_severity,
                .messageType = c.VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | c.VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT | c.VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT,
                .pfnUserCallback = vk_debug_callback,
            };

            vk_check(c.vkCreateDebugUtilsMessengerEXT.?(self.handle, &create_info, null, &self.debug_messenger));
        }

        return true;
    }

    fn vk_debug_callback(
        message_severity: c.VkDebugUtilsMessageSeverityFlagBitsEXT,
        message_types: c.VkDebugUtilsMessageTypeFlagsEXT,
        callback_data: [*c]const c.VkDebugUtilsMessengerCallbackDataEXT,
        user_data: ?*anyopaque,
    ) callconv(.c) u32 {
        _ = message_types;
        _ = user_data;

        const message = callback_data.*.pMessage;

        switch (message_severity) {
            c.VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT => logger.log_error(.ERROR, "{s}", .{message}),
            c.VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT => logger.log_debug(.WARNING, "{s}", .{message}),
            c.VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT => logger.log_debug(.INFO, "{s}", .{message}),
            else => {},
        }
        return c.VK_FALSE;
    }

    pub fn deinit(self: *Self) void {
        if (self.handle != null) {
            if (builtin.mode == .Debug) {
                c.vkDestroyDebugUtilsMessengerEXT.?(self.handle, self.debug_messenger, null);
            }
            c.vkDestroyInstance.?(self.handle, null);
            self.handle = null;
        }
    }
};

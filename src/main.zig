const std = @import("std");
const builtin = @import("builtin");
const c = @import("c.zig").c;
const vk_shared = @import("vk_shared.zig");
const vk_check = vk_shared.vk_check;
const window_mod = @import("window.zig");
const FixedArray = @import("fixed_array.zig").FixedArray;
const VulkanInstance = @import("vk_instance.zig").VulkanInstance;
const VulkanDevice = @import("vk_device.zig").VulkanDevice;
const VulkanSwapchain = @import("vk_swapchain.zig").VulkanSwapchain;
const Window = window_mod.Window;
const Application = @import("app.zig").Application;

pub fn main() !void {
    _ = vk_check(c.volkInitialize());

    var app: Application = try .init();
    defer app.deinit();

    var window: Window = .init(1920, 1080, "MAIN_WINDOW");
    defer window.deinit();

    var context: vk_shared.VulkanContext = undefined;
    defer context.deinit();

    if (!VulkanInstance.init(&context.instance)) {
        return;
    }

    window_mod.create_vk_surface(&context, window);
    if (!VulkanDevice.init(context.instance.handle, context.surface, &context.device)) {
        return;
    }

    const extent: c.VkExtent2D = .{ .width = 1024, .height = 840 };

    if (!VulkanSwapchain.init(&context.device, context.surface, extent, &context.swapchain)) {
        return;
    }
}

const c = @import("c.zig").c;
const device_mod = @import("vk_device.zig");
const std = @import("std");
const builtin = @import("builtin");
const VulkanInstance = @import("vk_instance.zig").VulkanInstance;
const VulkanDevice = device_mod.VulkanDevice;
const VulkanSwapchain = @import("vk_swapchain.zig").VulkanSwapchain;
const Application = @import("app.zig").Application;

pub const VulkanContext = struct {
    device: VulkanDevice,
    swapchain: VulkanSwapchain,
    instance: VulkanInstance,
    surface: c.VkSurfaceKHR,
    app: *Application,

    const Self = @This();

    pub fn init(app: *Application) Self {
        var context: VulkanContext = undefined;
        context.app = app;
        return context;
    }

    pub fn deinit(self: *Self) void {
        self.swapchain.deinit(self.device.logical_device);
        c.vkDestroySurfaceKHR.?(self.instance.handle, self.surface, null);
        self.device.deinit();
        self.instance.deinit();
    }
};

pub fn vk_check(r: c.VkResult) void {
    if (r != c.VK_SUCCESS) {
        if (builtin.mode == .Debug) {
            std.log.debug("Vk check failed, result was {d}", .{r});
        }
        @panic("vk_check failed!");
    }
}

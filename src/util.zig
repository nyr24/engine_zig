const std = @import("std");
const builtin = @import("builtin");

pub fn assert_msg(condition: bool, comptime fmt: []const u8, args: anytype) void {
    if (builtin.mode == .Debug or builtin.mode == .ReleaseSafe) {
        if (!condition) {
            std.debug.panic(fmt, args);
        }
    }
}

pub fn is_address_in_range(addr: [*]u8, start: [*]u8, offset: usize) bool {
    const addr_int = @intFromPtr(addr);
    const start_int = @intFromPtr(start);
    return (addr_int >= start_int) and (addr_int < (start_int + offset));
}

pub fn ptr_as_offset(addr: [*]u8, start: [*]u8) usize {
    const addr_int = @intFromPtr(addr);
    const start_int = @intFromPtr(start);
    std.debug.assert(addr_int >= start_int);
    return addr_int - start_int;
}

pub fn offset_as_ptr(offset: usize, start: [*]u8) [*]u8 {
    const start_int = @intFromPtr(start);
    return @ptrCast(start_int + offset);
}

const std = @import("std");
const builtin = @import("builtin");
const FixedArray = @import("fixed_array.zig").FixedArray;

const LogLevel = enum {
    FATAL,
    ERROR,
    WARNING,
    INFO,
    DEBUG,
    TRACE,
    TEST,
    _COUNT,
};

// FATAL,ERROR,WARN,INFO,DEBUG,TRACE,TEST
const color_strings: FixedArray([]const u8, @intFromEnum(LogLevel._COUNT)) = .init_from_slice(&[_][]const u8{
    "\x1b[0;41",
    "\x1b[1;31",
    "\x1b[1;33",
    "\x1b[1;32",
    "\x1b[1;34",
    "\x1b[1;28",
    "\x1b[45;37",
});

const log_level_as_str: FixedArray([]const u8, @intFromEnum(LogLevel._COUNT)) = .init_from_slice(&[_][]const u8{
    "[FATAL]: ",
    "[ERROR]: ",
    "[WARN]: ",
    "[INFO]: ",
    "[DEBUG]: ",
    "[TRACE]: ",
    "[TEST]: ",
});

fn log(lvl: LogLevel, comptime fmt: []const u8, args: anytype) void {
    var buff: [1256]u8 = undefined;
    var msg_buff: [1024]u8 = undefined;
    var writer = std.fs.File.stdout().writer(&buff);

    const message_fmt = std.fmt.bufPrint(&msg_buff, fmt, args) catch {
        std.log.err("bufPrint failed", .{});
        return;
    };

    writer.interface.print("\x1b[{s}m{s}{s}\x1b[0m\n", .{ color_strings.items[@intFromEnum(lvl)], log_level_as_str.items[@intFromEnum(lvl)], message_fmt }) catch {
        std.log.err("print failed", .{});
        return;
    };

    writer.interface.flush() catch {
        std.log.err("flush failed", .{});
        return;
    };
}

pub fn log_debug(lvl: LogLevel, comptime fmt: []const u8, args: anytype) void {
    if (builtin.mode == .Debug or builtin.mode == .ReleaseSafe) {
        log(lvl, fmt, args);
    }
}

// stays in release mode
pub fn log_error(lvl: LogLevel, comptime fmt: []const u8, args: anytype) void {
    log(lvl, fmt, args);
}

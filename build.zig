const std = @import("std");
const builtin = @import("builtin");
const Module = std.Build.Module;

const wayland_system_libs = [_][]const u8{"wayland-client"};
const x11_system_libs = [_][]const u8{ "xcb", "xcb-xkb", "X11", "X11-xcb" };
const win32_system_libs = [_][]const u8{};

const general_macros = [_][]const u8{"ENABLE_VULKAN_RENDERDOC_CAPTURE"};
const wayland_platform_macros = [_][]const u8{"VK_USE_PLATFORM_WAYLAND_KHR"};
const x11_platform_macros = [_][]const u8{"VK_USE_PLATFORM_XCB_KHR"};
const win32_platform_macros = [_][]const u8{"VK_USE_PLATFORM_WINDOWS_KHR"};

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const vulkan_sdk_path_env = std.process.getEnvVarOwned(b.allocator, "VULKAN_SDK") catch {
        @panic("VULKAN_SDK environment variable not set.");
    };
    defer b.allocator.free(vulkan_sdk_path_env);

    const linux_session = b.option([]const u8, "linux_session", "") orelse "wl";

    var build_dir: []const u8 = undefined;
    if (builtin.mode == .Debug) {
        build_dir = "build/debug";
    } else {
        build_dir = "build/release";
    }

    const root_mod = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });

    if (builtin.os.tag == .windows) {
        for (win32_system_libs) |sys_lib| {
            root_mod.linkSystemLibrary(sys_lib, .{});
        }
        for (win32_platform_macros) |macro| {
            root_mod.addCMacro(macro, "1");
        }
    } else if (builtin.os.tag == .linux) {
        if (std.mem.eql(u8, linux_session, "wl")) {
            for (wayland_system_libs) |sys_lib| {
                root_mod.linkSystemLibrary(sys_lib, .{});
            }
            for (wayland_platform_macros) |macro| {
                root_mod.addCMacro(macro, "1");
            }
        } else {
            for (x11_system_libs) |sys_lib| {
                root_mod.linkSystemLibrary(sys_lib, .{});
            }
            for (x11_platform_macros) |macro| {
                root_mod.addCMacro(macro, "1");
            }
        }
    }

    build_volk(b, root_mod) catch {
        std.log.err("Volk build failed", .{});
    };

    // glfw
    root_mod.addIncludePath(b.path("deps/glfw/include"));
    root_mod.addLibraryPath(b.path(b.pathJoin(&.{ build_dir, "glfw/src/" })));
    root_mod.linkSystemLibrary("glfw3", .{});

    // vulkan
    root_mod.addSystemIncludePath(.{ .cwd_relative = (b.pathResolve(&.{ vulkan_sdk_path_env, "include/vulkan" })) });
    root_mod.addLibraryPath(.{ .cwd_relative = (b.pathResolve(&.{ vulkan_sdk_path_env, "lib" })) });
    root_mod.linkSystemLibrary("vulkan_radeon", .{});

    const exe = b.addExecutable(.{
        .name = "engine",
        .root_module = root_mod,
        .use_llvm = true,
    });

    const installArtifact = b.addInstallArtifact(exe, .{ .dest_dir = .{
        .override = .{ .custom = "./" },
    } });

    b.getInstallStep().dependOn(&installArtifact.step);

    const run_step = b.step("run", "Run the app");

    const run_cmd = b.addRunArtifact(exe);
    run_step.dependOn(&run_cmd.step);

    run_cmd.step.dependOn(b.getInstallStep());

    if (b.args) |args| {
        run_cmd.addArgs(args);
    }
}

fn collect_c_sources(b: *std.Build, module: *Module, dir_path: std.Build.LazyPath, allowed_extensions: []const []const u8, recursive: bool) !void {
    const MAX_SRC_COUNT = 500;
    const cwd = std.fs.cwd();
    var dir = try cwd.openDir(dir_path.src_path.sub_path, .{ .iterate = true });
    defer std.fs.Dir.close(&dir);

    var sources: [MAX_SRC_COUNT][]const u8 = undefined;
    var i: usize = 0;

    if (!recursive) {
        var dir_it = std.fs.Dir.iterate(dir);

        while (try dir_it.next()) |entry| {
            const ext = std.fs.path.extension(entry.name);
            const include_file = for (allowed_extensions) |e| {
                if (std.mem.eql(u8, ext, e))
                    break true;
            } else false;

            if (include_file) {
                sources[i] = b.dupe(b.pathJoin(&.{ dir_path.src_path.sub_path, entry.name }));
                i += 1;
            }
        }
        module.addCSourceFiles(.{ .files = sources[0..i] });
    } else {
        var dir_walker = try std.fs.Dir.walk(dir, b.allocator);

        while (try dir_walker.next()) |entry| {
            const ext = std.fs.path.extension(entry.basename);
            const include_file = for (allowed_extensions) |e| {
                if (std.mem.eql(u8, ext, e)) {
                    break true;
                }
            } else false;

            if (include_file) {
                sources[i] = b.dupe(b.pathJoin(&.{ dir_path.src_path.sub_path, entry.basename }));
                i += 1;
            }
        }
        module.addCSourceFiles(.{ .files = sources[0..i] });
    }
}

fn build_volk(b: *std.Build, root_mod: *Module) !void {
    root_mod.addIncludePath(b.path("deps/volk/"));
    root_mod.addCSourceFile(.{ .file = b.path("deps/volk/volk.c") });
}

const std = @import("std");
const Module = std.Build.Module;

const Platform = enum { WAYLAND, X11, WIN32 };

const cmd_build_options = [_][]const u8{ "wl", "x11", "win32" };

const vk_platform_macros = [_][]const u8{
    "VK_USE_PLATFORM_WAYLAND_KHR",
    "VK_USE_PLATFORM_XCB_KHR",
    "VK_USE_PLATFORM_WINDOWS_KHR",
};

fn collect_c_sources(b: *std.Build, module: *Module, dir_path: std.Build.LazyPath, allowed_extensions: []const []const u8, recursive: bool) !void {
    // b.option(\, name_raw: []const u8, description_raw: []const u8)

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

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const vulkan_sdk_path_env = std.process.getEnvVarOwned(b.allocator, "VULKAN_SDK") catch {
        std.log.err("VULKAN_SDK environment variable not set.", .{});
        @panic("VULKAN_SDK not set");
    };
    defer b.allocator.free(vulkan_sdk_path_env);

    const root_mod = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });

    // TODO: should be configurable from command line arguments
    const c_macros = [_][]const u8{
        // "GLFW_INCLUDE_WAYLAND",
        "VK_USE_PLATFORM_WAYLAND_KHR",
        "ENABLE_VULKAN_RENDERDOC_CAPTURE",
    };

    for (c_macros) |macro| {
        root_mod.addCMacro(macro, "1");
    }

    build_volk(b, root_mod) catch {
        std.log.err("Volk build failed", .{});
    };

    // glfw
    root_mod.addIncludePath(b.path("deps/glfw/include"));
    root_mod.addLibraryPath(b.path("zig-out/bin/glfw/src/"));
    root_mod.linkSystemLibrary("glfw3", .{});

    // vulkan
    root_mod.addSystemIncludePath(.{ .cwd_relative = (b.pathResolve(&.{ vulkan_sdk_path_env, "include/vulkan" })) });
    root_mod.addLibraryPath(.{ .cwd_relative = (b.pathResolve(&.{ vulkan_sdk_path_env, "lib" })) });
    root_mod.linkSystemLibrary("vulkan_radeon", .{});

    root_mod.linkSystemLibrary("wayland-client", .{});

    const exe = b.addExecutable(.{
        .name = "game_engine",
        .root_module = root_mod,
        .use_llvm = true,
    });

    b.installArtifact(exe);

    const run_step = b.step("run", "Run the app");

    const run_cmd = b.addRunArtifact(exe);
    run_step.dependOn(&run_cmd.step);

    run_cmd.step.dependOn(b.getInstallStep());

    if (b.args) |args| {
        run_cmd.addArgs(args);
    }
}

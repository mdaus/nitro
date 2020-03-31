set(TARGET_NAME UUID)

# for Windows we use the UuidCreateSequential function
# (requires linking with Rpcrt4)

if (NOT WIN32)
    coda_add_driver(
        NAME ${TARGET_NAME}
        ARCHIVE "e2fsprogs-1.40-uuid.tar"
        HASH "SHA256=3507c7e9f226656b3903651c038e9c4907a31282cc4ba86aed8a588e6cc4cec2"
    )
endif()

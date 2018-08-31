set("TARGET_NAME"		"XercesC")

# Build as a completely external target
coda_add_driver("${TARGET_NAME}" 
	"xerces-c-3.1.1.tar" "SHA256=1f57a27dd1b3806df42d5ae638c362358cd16519578d7d395fc77ffd35f4dbcb"
	FALSE
)

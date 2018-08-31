set("TARGET_NAME"		"EXPAT")

# Build as a completely external target
coda_add_driver("${TARGET_NAME}" 
	"expat-2.0.0.tar" "SHA256=57ecb912d6ebc131665069a13a832305b011c9e7990b66bcc9457439aec610ac" 
	FALSE
)

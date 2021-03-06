set(java_source_dir ${CMAKE_CURRENT_SOURCE_DIR}/src/java/nitf)
set(java_files ${java_source_dir}/BandInfo.java
               ${java_source_dir}/BandSource.java
               ${java_source_dir}/BlockingInfo.java
               ${java_source_dir}/CloneableObject.java
               ${java_source_dir}/ComponentInfo.java
               ${java_source_dir}/DESegment.java
               ${java_source_dir}/DestructibleObject.java
               ${java_source_dir}/DESubheader.java
               ${java_source_dir}/DownSampler.java
               ${java_source_dir}/Extensions.java
               ${java_source_dir}/Field.java
               ${java_source_dir}/FieldType.java
               ${java_source_dir}/FieldWarning.java
               ${java_source_dir}/FileHeader.java
               ${java_source_dir}/FileSecurity.java
               ${java_source_dir}/FileSource.java
               ${java_source_dir}/GraphicSegment.java
               ${java_source_dir}/GraphicSubheader.java
               ${java_source_dir}/ImageReader.java
               ${java_source_dir}/ImageSegment.java
               ${java_source_dir}/ImageSource.java
               ${java_source_dir}/ImageSubheader.java
               ${java_source_dir}/ImageWriter.java
               ${java_source_dir}/IOHandle.java
               ${java_source_dir}/IOInterface.java
               ${java_source_dir}/LabelSegment.java
               ${java_source_dir}/LabelSubheader.java
               ${java_source_dir}/LookupTable.java
               ${java_source_dir}/MaxDownSampler.java
               ${java_source_dir}/MemoryDestructor.java
               ${java_source_dir}/MemoryIO.java
               ${java_source_dir}/MemorySource.java
               ${java_source_dir}/NativeIOInterface.java
               ${java_source_dir}/NITFException.java
               ${java_source_dir}/NITFObject.java
               ${java_source_dir}/NITFResourceManager.java
               ${java_source_dir}/PixelSkipDownSampler.java
               ${java_source_dir}/PluginRegistry.java
               ${java_source_dir}/Reader.java
               ${java_source_dir}/Record.java
               ${java_source_dir}/RESegment.java
               ${java_source_dir}/RESubheader.java
               ${java_source_dir}/SegmentReader.java
               ${java_source_dir}/SegmentSource.java
               ${java_source_dir}/SegmentWriter.java
               ${java_source_dir}/StreamIOWriteHandler.java
               ${java_source_dir}/SubWindow.java
               ${java_source_dir}/SumSq2BandDownSampler.java
               ${java_source_dir}/TextSegment.java
               ${java_source_dir}/TextSubheader.java
               ${java_source_dir}/TRE.java
               ${java_source_dir}/Version.java
               ${java_source_dir}/WriteHandler.java
               ${java_source_dir}/Writer.java)

set(jni_source_dir ${CMAKE_CURRENT_SOURCE_DIR}/src/jni/source)
set(c_files ${jni_source_dir}/nitf_BandInfo.c
            ${jni_source_dir}/nitf_BandSource.c
            ${jni_source_dir}/nitf_BlockingInfo.c
            ${jni_source_dir}/nitf_ComponentInfo.c
            ${jni_source_dir}/nitf_DESegment.c
            ${jni_source_dir}/nitf_DESubheader.c
            ${jni_source_dir}/nitf_DownSampler.c
            ${jni_source_dir}/nitf_Extensions.c
            ${jni_source_dir}/nitf_Extensions_ExtensionsIterator.c
            ${jni_source_dir}/nitf_Field.c
            ${jni_source_dir}/nitf_FieldWarning.c
            ${jni_source_dir}/nitf_FileHeader.c
            ${jni_source_dir}/nitf_FileSecurity.c
            ${jni_source_dir}/nitf_FileSource.c
            ${jni_source_dir}/nitf_GraphicSegment.c
            ${jni_source_dir}/nitf_GraphicSubheader.c
            ${jni_source_dir}/nitf_ImageReader.c
            ${jni_source_dir}/nitf_ImageSegment.c
            ${jni_source_dir}/nitf_ImageSource.c
            ${jni_source_dir}/nitf_ImageSubheader.c
            ${jni_source_dir}/nitf_ImageWriter.c
            ${jni_source_dir}/nitf_IOHandle.c
            ${jni_source_dir}/nitf_IOInterface.c
            ${jni_source_dir}/nitf_JNI.c
            ${jni_source_dir}/nitf_LabelSegment.c
            ${jni_source_dir}/nitf_LabelSubheader.c
            ${jni_source_dir}/nitf_LookupTable.c
            ${jni_source_dir}/nitf_MaxDownSampler.c
            ${jni_source_dir}/nitf_MemorySource.c
            ${jni_source_dir}/nitf_NativeIOInterface.c
            ${jni_source_dir}/nitf_PixelSkipDownSampler.c
            ${jni_source_dir}/nitf_PluginRegistry.c
            ${jni_source_dir}/nitf_Reader.c
            ${jni_source_dir}/nitf_Record.c
            ${jni_source_dir}/nitf_RESegment.c
            ${jni_source_dir}/nitf_RESubheader.c
            ${jni_source_dir}/nitf_SegmentReader.c
            ${jni_source_dir}/nitf_SegmentSource.c
            ${jni_source_dir}/nitf_SegmentWriter.c
            ${jni_source_dir}/nitf_SubWindow.c
            ${jni_source_dir}/nitf_SumSq2BandDownSampler.c
            ${jni_source_dir}/nitf_TextSegment.c
            ${jni_source_dir}/nitf_TextSubheader.c
            ${jni_source_dir}/nitf_TRE.c
            ${jni_source_dir}/nitf_TRE_TREIterator.c
            ${jni_source_dir}/nitf_WriteHandler.c
            ${jni_source_dir}/nitf_Writer.c)

add_java_module(nitf "${java_files}" "${c_files}" "${coda_jars}")
add_dependencies(nitf-java nitf-c)


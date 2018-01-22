import nitro
import numpy as np
import unittest


class TestDes(unittest.TestCase):
    def test_des_create(self):
        record = nitro.Record()
        d1 = record.new_data_extension_segment()
        d1.subheader.NITF_DE = "DE"
        d1.subheader.NITF_DESTAG = "TEST DES"
        d1.subheader.NITF_DESVER = "01"
        # d1.subheader.NITF_DESCLAS = "U"
        t = d1.subheader.subheaderFields = nitro.TRE("TEST DES", "TEST DES")
        t.TEST_DES_COUNT = "16"
        t.TEST_DES_START = "065"
        t.TEST_DES_INCREMENT = "01"

        data = np.frombuffer(b"123456789ABCDEF0", dtype='uint8')

        with nitro.IOHandle("test.ntf", nitro.AccessFlags.NITF_ACCESS_WRITEONLY, nitro.CreationFlags.NITF_CREATE) as handle:
            with nitro.Writer(record, handle) as writer:
                deswriter = writer.new_data_extension_writer(0)
                desdata = nitro.SegmentMemorySource(data)
                deswriter.attach_source(desdata)


if __name__ == '__main__':
    unittest.main()


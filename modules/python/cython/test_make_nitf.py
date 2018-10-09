#!/usr/bin/env python -O
# -*- coding: utf-8 -*-

"""
 * =========================================================================
 * This file is part of NITRO
 * =========================================================================
 *
 * (C) Copyright 2004 - 2016, MDA Information Systems LLC
 *
 * NITRO is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this program; if not, If not,
 * see <http://www.gnu.org/licenses/>.
 *
 *
"""

"""
This is an example of how to generate your own NITF files
from scratch

You could also read in a JPG file, creating a NITF with the
decompressed raw bytes, using the PIL library.

Notice how easy it is to change fields in the NITF
"""
import os, sys, array, unittest, random
import numpy
import struct
from nitro import *
from tempfile import mkstemp
from datetime import datetime
import logging

logging.basicConfig(level=logging.DEBUG, stream=sys.stdout,
                    format='%(asctime)s %(levelname)s %(message)s')

NITF_21_DATE_FORMAT = '%Y%m%d%H%M%S'


class TestCreator(unittest.TestCase):

    def setUp(self):
        """ just sets up mappings of default field values """
        fieldmap = {}
        fieldmap['header'] = {
            'fileHeader': 'NITF',
            'fileVersion': '02.10',
            'systemType': 'BF01',
            'classification': 'U',
            'originatorName': 'Chuck Norris',
            'complianceLevel': 99,
            'encrypted': 0,
            'originStationID': 'MRDC',
            'fileDateTime': datetime.now().strftime(NITF_21_DATE_FORMAT),
        }
        fieldmap['image'] = {
            'filePartType': 'IM',
            'classificationSystem': 'U',
            'imageDateAndTime': datetime.now().strftime(NITF_21_DATE_FORMAT),
        }

        fieldmap['graphic'] = {
            'filePartType': 'SY',
            'securityClass': 'U',
            'encrypted': 0,
            'color': 'C',
            'stype': 'C',
        }
        self.fieldmap = fieldmap

        self.inputs = sys.argv[1:]
        self.hasInput = len(self.inputs) > 0

    def test_make_image_nitf(self):
        self.make_image_nitf()

    def make_image_nitf(self, numBands=2, width=256, height=256):
        """ makes a nitf, with artificial image data """
        record = Record()
        if 'header' in self.fieldmap:
            for field, value in self.fieldmap['header'].items():
                if field in record.header:
                    record.header[field] = value

        # create a random pattern size
        csize = random.randint(0, 255)
        alldata = []
        for i in range(height):
            for j in range(width):
                for k in range(numBands):
                    swap = (j % csize) % 2 != 0
                    alldata.append((j % csize) % 2 == 0 and (not swap and 255 or 0))

        # add an image segment
        segment = record.newImageSegment()
        if 'image' in self.fieldmap:
            for field, value in self.fieldmap['image'].items():
                if field in segment.subheader:
                    segment.subheader[field] = value

        # for now, I just put in these as defaults
        # obviously, too large of an image will cause problems with the block-size...
        segment.addBands(numBands)
        segment.subheader['numrows'] = height
        segment.subheader['numcols'] = width
        segment.subheader['numBitsPerPixel'] = 16
        segment.subheader['actualBitsPerPixel'] = 16
        segment.subheader['pixelValueType'] = 'INT'
        segment.subheader['pixelJustification'] = 'R'
        segment.subheader['imageCompression'] = 'NC'
        segment.subheader['imageDisplayLevel'] = 1
        segment.subheader['imageAttachmentLevel'] = 0
        segment.subheader['imageRepresentation'] = numBands == 1 and 'MONO' or 'MULTI'
        segment.subheader['imageCategory'] = 'VIS'
        segment.subheader['imageSyncCode'] = '0'
        segment.subheader['imageLocation'] = '0000000000'
        segment.subheader['imageMode'] = 'B'
        segment.subheader['numBlocksPerCol'] = 1
        segment.subheader['numBlocksPerRow'] = 1
        segment.subheader['numPixelsPerHorizBlock'] = width
        segment.subheader['numPixelsPerVertBlock'] = height
        segment.subheader['numImageComments'] = '2'
        segment.subheader.imageComments.append(Field(value="this is a comment"))
        segment.subheader.imageComments.append(Field(value="another comment"))

        alldata = numpy.ascontiguousarray(alldata, dtype="int16")
        tmp = alldata
        tmp.shape = (height, width, numBands)

        pixmta = TRE('PIXMTA')
        pixmta.setField('NUMAIS', '001')
        pixmta.setField('AISDLVL[0]', '001')
        pixmta.setField('ORIGIN_X', '+0.0000000E+00')
        pixmta.setField('ORIGIN_Y', '+0.0000000E+00')
        pixmta.setField('SCALE_X', '+1.0000000E+00')
        pixmta.setField('SCALE_Y', '+1.0000000E+00')
        pixmta.setField('SAMPLE_MODE', '1')
        pixmta.setField('NUMMETRICS', '00002')
        pixmta.setField('PERBAND', 'P')
        pixmta.setField('DESCRIPTION[0]', 'SPECTRAL BAND-SPECIFIC SMEAR MAGNITUDE')
        pixmta.setField('UNIT[0]', 'Pixels')
        pixmta.setField('FITTYPE[0]', 'D')
        pixmta.setField('DESCRIPTION[1]', 'SPECTRAL BAND-SPECIFIC SMEAR DIRECTION')
        pixmta.setField('UNIT[1]', 'deg')
        pixmta.setField('FITTYPE[1]', 'D')
        pixmta.setField('RESERVED_LEN', '00000')

        import uuid
        csexrb = TRE('CSEXRB')
        csexrb.setField('IMAGE_UUID', str(uuid.uuid4()))
        csexrb.setField('NUM_ASSOC_DES', '001')
        csexrb.setField('ASSOC_DES_ID[0]', str(uuid.uuid4()))
        csexrb.setField('PLATFORM_ID', 'PLATFM')
        csexrb.setField('PAYLOAD_ID', 'PAYLD')
        csexrb.setField('SENSOR_ID', 'SENSR')
        csexrb.setField('SENSOR_TYPE', 'F')
        csexrb.setField('GROUND_REF_POINT_X', '+00000000.00')
        csexrb.setField('GROUND_REF_POINT_Y', '+00000000.00')
        csexrb.setField('GROUND_REF_POINT_Z', '+00000100.00')
        csexrb.setField('TIME_STAMP_LOC', '0')
        csexrb.setField('REFERENCE_FRAME_NUM', '000000000')
        csexrb.setField('BASE_TIMESTAMP', '20170101000000.000000000')
        csexrb.setField('DT_MULTIPLIER', struct.pack('>Q', int(2e9)))

        xhd = segment.subheader.getXHD()
        xhd.append(pixmta)
        xhd.append(csexrb)

        # setup image source
        imagesource = ImageSource()
        for i in range(numBands):
            imagesource.addBand(MemoryBandSource(alldata, len(alldata) / numBands, i, 1, numBands - 1))

        try:
            handle, outfile = mkstemp(dir=os.getcwd(), suffix='.ntf')
            os.close(handle)
            writer = Writer()
            outhandle = IOHandle(outfile, IOHandle.ACCESS_WRITEONLY, IOHandle.CREATE)
            if writer.prepare(record, outhandle):
                imagewriter = writer.newImageWriter(0)
                if imagewriter:
                    imagewriter.attachSource(imagesource)
                    writer.write()
                    logging.info('Successfully Wrote NEW NITF: %s\n' % outfile)
            outhandle.close()
        except Exception as e:
            print(e)
        else:
            handle = IOHandle(outfile)
            reader = Reader()
            record = reader.read(handle)
            logging.info(str(record.header))

            # Read back in the band data to make sure the read/write
            # went okay
            subheader = record.getImages()[0].subheader
            imageReader = reader.newImageReader(0)

            window = SubWindow()
            window.numRows = subheader['numRows'].intValue()
            window.numCols = subheader['numCols'].intValue()
            window.bandList = list(range(subheader.getBandCount()))
            nbpp = subheader['numBitsPerPixel'].intValue()
            bandData = imageReader.read(window, dtype=np.int16)
            readData = []
            for item in bandData[0]:
                readData.append(item)
            # assert (readData == alldata).all()

            handle.close()

            import subprocess
            subprocess.run(['show_nitf', outfile])
            try:
                os.unlink('test.ntf')
            except:
                pass
            os.link(outfile, 'test.ntf')


if __name__ == '__main__':
    unittest.main(argv=sys.argv[0:1])

import unittest
try:
    from replay import evidence_key
except ImportError:
    evidence_key=None

class CacheKeyTest(unittest.TestCase):
    def test_different_image_region_or_engine_cannot_reuse_context(self):
        self.assertIsNotNone(evidence_key,'evidence cache key missing')
        base=evidence_key('image-a',[10,20,180,36],'engine-a')
        self.assertNotEqual(base,evidence_key('image-b',[10,20,180,36],'engine-a'))
        self.assertNotEqual(base,evidence_key('image-a',[11,20,180,36],'engine-a'))
        self.assertNotEqual(base,evidence_key('image-a',[10,20,180,36],'engine-b'))

if __name__=='__main__':unittest.main()

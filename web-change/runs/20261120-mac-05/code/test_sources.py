import unittest
from source_utils import sanitize,git_blob_sha

class SourceContracts(unittest.TestCase):
    def test_network_assets_and_scripts_removed_but_content_remains(self):
        html='<html><head><link rel="stylesheet" href="https://x/a.css"><script>alert(1)</script><style>h1{color:red}</style></head><body><h1>Source heading</h1><p>Original source text</p><img src="https://x/a.png"><iframe src="https://x"></iframe></body></html>'
        clean,audit=sanitize(html,'@import url(https://x/b.css); p{background:url("https://x/a.png");color:blue}')
        self.assertIn('Original source text',clean)
        self.assertIn('color:blue',clean)
        self.assertNotIn('<script',clean)
        self.assertNotIn('<img',clean)
        self.assertNotIn('<iframe',clean)
        self.assertNotIn('https://x',clean)
        self.assertGreaterEqual(audit['removed_tags'],4)

    def test_git_blob_hash_uses_git_header(self):
        self.assertEqual(git_blob_sha(b'test content\n'),'d670460b4b4aece5915caf5c68d12f560a9fe3e4')

    def test_quoted_svg_url_parentheses_do_not_corrupt_css(self):
        css=':root{--icon:url("data:image/svg+xml,<svg fill=\'rgb(1,2,3)\'></svg>");--color:blue}p{color:red}'
        clean,_=sanitize('<html><head></head><body><p>Original text</p></body></html>',css)
        self.assertNotIn('<svg',clean)
        self.assertNotIn('rgb(1,2,3)',clean)
        self.assertNotIn('</svg>',clean)
        self.assertIn('--icon:none',clean)
        self.assertIn('--color:blue',clean)
        self.assertIn('p{color:red}',clean)

    def test_inline_declaration_keeps_non_asset_properties(self):
        clean,_=sanitize('<html><head></head><body><p style="background:url(https://example.com/a.png);color:red">Visible text</p></body></html>','')
        self.assertIn('color:red',clean)
        self.assertNotIn('example.com',clean)

if __name__=='__main__':unittest.main()

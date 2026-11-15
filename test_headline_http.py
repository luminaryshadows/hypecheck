import threading,unittest,urllib.request,urllib.error
from http.server import ThreadingHTTPServer
from pathlib import Path
import serve
class HTTPTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.server=ThreadingHTTPServer(('127.0.0.1',0),serve.Handler)
  cls.thread=threading.Thread(target=cls.server.serve_forever,daemon=True);cls.thread.start()
  cls.base='http://127.0.0.1:'+str(cls.server.server_port)
 @classmethod
 def tearDownClass(cls):cls.server.shutdown();cls.server.server_close();cls.thread.join()
 def test_assets_and_strict_policy(self):
  for p in ['','headlines.html','headlines.js','headlines.css','headline-report.json','headline-predictions.json','headline-context.json','archive.html']:
   with urllib.request.urlopen(self.base+'/'+p) as r:
    self.assertEqual(r.status,200);self.assertNotIn('unsafe-inline',r.headers['Content-Security-Policy'])
    self.assertTrue(r.read())
 def test_no_inline_script_or_style(self):
  s=(Path(__file__).resolve().parent/'docs/index.html').read_text()
  self.assertNotIn('<style>',s);self.assertNotIn('<script>',s)
  self.assertIn('src="headlines.js"',s);self.assertIn('href="headlines.css"',s)
 def test_private_paths_stay_inaccessible(self):
  for path in ['/.git/config','/../README.md','/headline_study.py']:
   with self.assertRaises(urllib.error.HTTPError) as e:urllib.request.urlopen(self.base+path)
   self.assertEqual(e.exception.code,404);e.exception.close()
if __name__=='__main__':unittest.main()

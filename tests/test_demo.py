"""The local demo serves bundled UI and rejects unexpected execution inputs."""
import http.client
import json
import threading
import unittest
from http.server import ThreadingHTTPServer
from medadaptagent.demo.server import Handler

class DemoServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(('127.0.0.1',0),Handler)
        cls.worker=threading.Thread(target=cls.server.serve_forever,daemon=True)
        cls.worker.start()
    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown();cls.server.server_close();cls.worker.join()
    def request(self,method='GET',path='/',data=None,headers=None):
        conn=http.client.HTTPConnection('127.0.0.1',self.server.server_port,timeout=5)
        conn.request(method,path,body=data,headers=headers or {})
        r=conn.getresponse();status,body=r.status,r.read();conn.close();return status,body
    def test_bundled_page_is_served(self):
        code,body=self.request();self.assertEqual(code,200);self.assertIn(b'Try the tiny lab',body)
    def test_unknown_routes_do_not_expose_files(self):
        self.assertEqual(self.request(path='/../../.env')[0],404)
    def test_browser_from_other_origin_cannot_trigger_run(self):
        self.assertEqual(self.request('POST','/api/run',b'{}',{'Content-Type':'application/json','Origin':'https://example.org'})[0],403)
    def test_bad_inputs_fail_before_workflow(self):
        for params in [{'seed':True},{'shift':100},{'seed':-1},{'checkpoint':'private.pt'},{'dataset':'unknown'},{'sample':3}]:
            self.assertEqual(self.request('POST','/api/run',json.dumps(params),{'Content-Type':'application/json'})[0],400)
    def test_no_extra_fields_or_image_paths_accepted(self):
        code,body=self.request('POST','/api/run',b'{"path":"/private/file"}',{'Content-Type':'application/json'})
        self.assertEqual(code,400);self.assertNotIn(b'/private/file',body)

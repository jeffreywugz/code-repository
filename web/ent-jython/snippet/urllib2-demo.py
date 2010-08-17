#!/usr/bin/env jython
import java
import MyProvider
import httplib
import urllib2

# Install the all-trusting trust manager
java.security.Security.addProvider(MyProvider())
java.security.Security.setProperty("ssl.TrustManagerFactory.algorithm", "TrustAllCertificates")

http_proxy = '10.10.44.251:6588'
https_proxy = '10.10.44.251:6588'
user_agent = "Mozilla/5.0 (Windows; U; Windows NT 5.2; en-US; rv:1.9.1.8) Gecko/20100202 Firefox/3.5.8 GTB6 (.NET CLR 3.5.30729)"
headers = {
    'User-Agent': user_agent,
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'en-us,en;q=0.5'
    }

proxy_handler = urllib2.ProxyHandler({'http': http_proxy, 'https': https_proxy})
proxy_auth_handler = urllib2.ProxyBasicAuthHandler()
# proxy_auth_handler.add_password('realm', 'host', 'username', 'password')
# cookie=cookielib.CookieJar()
opener = urllib2.build_opener(proxy_handler, proxy_auth_handler)

req = urllib2.Request('https://mail.google.com/mail', data=None, headers=headers)
response = opener.open(req)
out = open('index.html', 'w')
out.write(response.read())

#!/usr/bin/python2
"""
Basic entry-points for metaWeblog API: see <http://www.xmlrpc.com/metaWeblogApi>

Signatures(mimic C language):
w--------------------------------------------------------------------------------
string metaWeblog.newPost(str blogid, str username, str password, dict struct, bool publish);
bool metaWeblog.editPost(str postid, str username, str password, dict struct, bool publish);
struct metaWeblog.getPost(str postid, str username, str password);
--------------------------------------------------------------------------------

some explanations:
--------------------------------------------------------------------------------
str blogid: Unique identifier of the blog the post will be added to.
bool publish: If true, the blog will be published immediately after the post is made. 
dict struct: a Python dict to define a post. The defined members of struct are the elements of <item> in RSS 2.0
   The three basic elements are title, link and description.
   For categories, pass an array of strings of names of categories that the post belongs to,
   named categories. On the server side, it's not an error if the category doesn't exist,
   only record categories for ones that do exist.
   see <http://cyber.law.harvard.edu/rss/rss.html#hrelementsOfLtitemgt>
--------------------------------------------------------------------------------
"""

import xmlrpclib
cnblogs = xmlrpclib.ServerProxy("http://www.cnblogs.com/ans42/services/metaweblog.aspx")
blog_id, user, passwd = 'ans42', 'ans42', 'iambook11'

def introspect(server):
    methods = server.system.listMethods()
    print methods
    methods = [(m, server.system.methodSignature(m), server.system.methodHelp(m)) for m in methods]
    return '\n'.join(['%s:%s\n%s'%(m, sig, help) for m, sig, help in methods])

metaWeblog = cnblogs.metaWeblog
print metaWeblog.getRecentPosts(blog_id, user, passwd, 1)

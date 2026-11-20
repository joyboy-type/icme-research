from hashlib import sha1
import re
from bs4 import BeautifulSoup
import tinycss2
from tinycss2.ast import IdentToken

def offline_css(text,declaration=False):
    def walk(tokens):
        cleaned=[]
        for token in tokens:
            if token.type=='at-rule' and token.lower_at_keyword in ['import','font-face']:continue
            if token.type=='url' or (token.type=='function' and token.lower_name=='url'):
                cleaned.append(IdentToken(token.source_line,token.source_column,'none'));continue
            if hasattr(token,'content') and token.content is not None:token.content=walk(token.content)
            if hasattr(token,'arguments'):token.arguments=walk(token.arguments)
            cleaned.append(token)
        return cleaned
    tokens=tinycss2.parse_component_value_list(text) if declaration else tinycss2.parse_stylesheet(text,skip_comments=False,skip_whitespace=False)
    return tinycss2.serialize(walk(tokens))

def sanitize(html,css):
    html=re.sub(r'\{\{.*?\}\}','',html,flags=re.S)
    soup=BeautifulSoup(html,'lxml');removed=0
    for node in list(soup.find_all(['script','noscript','link','img','picture','video','audio','iframe','canvas','source','object','embed'])):
        node.decompose();removed+=1
    for node in list(soup.find_all('meta',attrs={'http-equiv':True})):node.decompose();removed+=1
    for tag in soup.find_all(True):
        for key in list(tag.attrs):
            if key.lower().startswith('on') or key in ['autofocus','src','srcset']:del tag[key]
        if tag.has_attr('style'):tag['style']=offline_css(tag['style'],declaration=True)
    for style in soup.find_all('style'):style.string=offline_css(style.get_text())
    new=soup.new_tag('style');new.string=offline_css(css)+'\nhtml{scroll-behavior:auto!important}*,*::before,*::after{animation:none!important;transition:none!important;caret-color:transparent!important}'
    if soup.head:soup.head.append(new)
    else:soup.insert(0,new)
    return str(soup),{'removed_tags':removed,'sanitization':'Offline static source derivative; external assets/scripts removed; source CSS and embedded styles retained.'}

def git_blob_sha(data):
    return sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()

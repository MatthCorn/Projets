import xml.etree.ElementTree as ET
from xml.dom import minidom
import ast

'''Ensemble des outils de traduction en xml'''

#######################################################################################################
### Convertion d'un objet python en un XML avec une structure simple                                ###
### exemple :                                                                                       ###
###     objet :                                                                                     ###
###         dic = {'x': [1, 2, {'t': 'dico'}, [43, 52]], 'y': {'z': 2}}                             ###
###     XML correspondant :                                                                         ###
###         <?xml version="1.0" ?>                                                                  ###
###         <Dictionary>                                                                            ###
###          <List Key="x">                                                                         ###
###           <Object Value="1"/>                                                                   ###
###           <Object Value="2"/>                                                                   ###
###           <Dictionary>                                                                          ###
###            <Object Key="t" Value="dico"/>                                                       ###
###           </Dictionary>                                                                         ###
###           <List>                                                                                ###
###            <Object Value="43"/>                                                                 ###
###            <Object Value="52"/>                                                                 ###
###           </List>                                                                               ###
###          </List>                                                                                ###
###          <Dictionary Key="y">                                                                   ###
###           <Object Key="z" Value="2"/>                                                           ###
###          </Dictionary>                                                                          ###
###         </Dictionary>                                                                           ###
#######################################################################################################

def fromObjToXml(Obj):
    tree = ET.ElementTree()
    if type(Obj) == dict:
        root = ET.Element("Dictionary")
        for head, tail in Obj.items():
            if type(tail) == dict:
                ElType = "Dictionary"
            elif type(tail) == list:
                ElType = "List"
            elif type(tail) == str:
                ElType = "String"
            else:
                ElType = "Object"

            El = ET.Element(ElType)
            fromObjToXmlRec(El, tail, head)
            root.append(El)
    elif type(Obj) == list:
        root = ET.Element("List")
        for x in Obj:
            if type(x) == dict:
                ElType = "Dictionary"
            elif type(x) == list:
                ElType = "List"
            elif type(x) == str:
                ElType = "String"
            else:
                ElType = "Object"
            El = ET.Element(ElType)
            fromObjToXmlRec(El, x)
            root.append(El)
    tree._setroot(root)
    return tree
        
def fromObjToXmlRec(Base, Tail, Head=None):
    if Head != None:
        Base.attrib['Key'] = str(Head)
    if type(Tail) == dict:
        for head, tail in Tail.items():
            if type(tail) == dict:
                ElType = "Dictionary"
            elif type(tail) == list:
                ElType = "List"
            elif type(tail) == str:
                ElType = "String"
            else:
                ElType = "Object"
            El = ET.Element(ElType)
            fromObjToXmlRec(El, tail, head)
            Base.append(El)

    elif type(Tail) == list:
        for x in Tail:
            if type(x) == dict:
                ElType = "Dictionary"
            elif type(x) == list:
                ElType = "List"
            elif type(x) == str:
                ElType = "String"
            else:
                ElType = "Object"


            El = ET.Element(ElType)
            fromObjToXmlRec(El, x)
            Base.append(El)

    else:
        Base.attrib['Value'] = str(Tail)


def saveObjAsXml(dic, file):
    tree = fromObjToXml(dic)
    rought = ET.tostring(tree.getroot(), 'utf-8')
    parsed = minidom.parseString(rought)
    new = parsed.toprettyxml(indent=" ")
    outfile = open(file, "w")
    outfile.write(new)
    outfile.close()

#################################################################
### Convertion d'un XML à structure simple en un objet python ###
#################################################################

def fromXmlToObj(tree):
    if tree._root.tag == 'Dictionary':
        Obj = {}
        for El in tree.findall("./"):
            (El != tree) and Obj.update({El.get('Key'): fromXmlToObjRec(El)})
    elif tree._root.tag == 'List':
        Obj = []
        for El in tree.findall("./"):
            (El != tree) and Obj.append(fromXmlToObjRec(El))
    return Obj

def fromXmlToObjRec(Base):
    if Base.tag == 'String':
        return Base.get('Value')
    elif Base.tag == 'List':
        li = []
        for El in Base.findall("./"):
            (El != Base) and li.append(fromXmlToObjRec(El))
        return li
    elif Base.tag == 'Dictionary':
        dic = {}
        for El in Base.findall("./"):
            (El != Base) and dic.update({El.get('Key'): fromXmlToObjRec(El)})
        return dic
    elif Base.get('Value') == "nan":
        return float("nan")
    else:
        return ast.literal_eval(Base.get('Value'))
def loadXmlAsObj(file):
    tree = ET.parse(file)
    dic = fromXmlToObj(tree)
    return dic

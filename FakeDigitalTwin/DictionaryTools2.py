import copy
import xml.etree.ElementTree as ET
from xml.dom import minidom

'''Ensemble des outils relatifs aux dictionnaires (et accéssoirement aux tracductions en xml)'''


# Fusionne une liste de dictionnaire et renvoie un unique dictionnaire
def merge_dicts(dict_list):
    result = {}
    for dictionary in dict_list:
        result.update(dictionary)
    return result

# à partir d'un dictionnaire de sous-dictionnaires initial, crée un nouveau dictionnaire correspondant
# à la fusion des sous-dictionnaires auquels l'attribut "pere" a été ajouté, et correspond à la clé du 
# sous-dictionnaire dans le dictionnaire initial
def key_to_arg(dictionary):
    local_dict = copy.deepcopy(dictionary)
    for key in local_dict.keys():
        for WF in local_dict[key].keys():
            local_dict[key][WF]['pere'] = key
    return merge_dicts(list(local_dict.values()))


#######################################################################################################
### Convertion d'un dictionnaire en un XML avec une structure simple                                ###
### exemple :                                                                                       ###
###     dictionnaire :                                                                              ###
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

def fromDicToXml(Dictionary):
    tree = ET.ElementTree()
    root = ET.Element("Dictionary")
    for head,tail in Dictionary.items():
        if type(tail) == dict:
            El = ET.Element("Dictionary")
            fromDicToXmlRec(El,tail,head)
            root.append(El)
        elif type(tail) == list:
            El = ET.Element("List")
            fromDicToXmlRec(El,tail,head)
            root.append(El)
        else:
            El = ET.Element("Object")
            fromDicToXmlRec(El,tail,head)
            root.append(El)
    tree._setroot(root)
    return tree
        
def fromDicToXmlRec(Base,Tail,Head=None):
    if Head != None:
        Base.attrib['Key'] = str(Head)
    if type(Tail) == dict:
        for head,tail in Tail.items():
            if type(tail) == dict:
                El = ET.Element("Dictionary")
                fromDicToXmlRec(El,tail,head)
                Base.append(El)
            elif type(tail) == list:
                El = ET.Element("List")
                fromDicToXmlRec(El,tail,head)
                Base.append(El)
            else:
                El = ET.Element("Object")
                fromDicToXmlRec(El,tail,head)
                Base.append(El)
    elif type(Tail) == list:
        for x in Tail:
            if type(x) == dict:
                El = ET.Element("Dictionary")
                fromDicToXmlRec(El,x)
                Base.append(El)
            elif type(x) == list:
                El = ET.Element("List")
                fromDicToXmlRec(El,x)
                Base.append(El)
            else:
                El = ET.Element("Object")
                fromDicToXmlRec(El,x)
                Base.append(El)
    else:
        Base.attrib['Value'] = str(Tail)


def saveDicAsXml(dic,file):
    tree = fromDicToXml(dic)
    rought = ET.tostring(tree.getroot(), 'utf-8')
    parsed = minidom.parseString(rought)
    new = parsed.toprettyxml(indent=" ")
    outfile=open(file, "w")
    outfile.write(new)
    outfile.close()

#################################################################
### Convertion d'un XML à structure simple en un dictionnaire ###
#################################################################

def fromXmlToDic(tree):
    dic = {}
    for El in tree.findall("./"):
        (El != tree) and dic.update({El.get('Key') : fromXmlToDicRec(El)})
    return dic

def fromXmlToDicRec(Base):
    if Base.tag == 'Object':
        return Base.get('Value')
    elif Base.tag == 'List':
        li = []
        for El in Base.findall("./"):
            (El != Base) and li.append(fromXmlToDicRec(El))
        return li
    elif Base.tag == 'Dictionary':
        dic = {}
        for El in Base.findall("./"):
            (El != Base) and dic.update({El.get('Key') : fromXmlToDicRec(El)})
        return dic

def loadXmlAsDic(file):
    tree = ET.parse(file)
    dic = fromXmlToDic(tree)
    return dic


#########################################################################################
### Modification du fichier ThreatCatalog.xml en accord avec un dictionnaire cohérent ###
#########################################################################################

def modifXML(file,dic):
    data = loadXmlAsDic(file)
    newRadar = list(dic.keys())[0]
    newWFname = list(dic[newRadar].keys())[0]
    # On modifie toutes les itérations de la forme d'onde dans le dict
    for radar in data.values():
        if newWFname in radar.keys():
            radar.update(dic[newRadar])
    # On modifie le radar s'il existe (quitte à le faire pour une deuxième fois s'il possédait la WF)
    if newRadar in data.values():
        data[newRadar].update(dic[newRadar])
    # Dans le cas contraire on ajoute le nouveau radar avec sa WF
    else:
        data.update(dic)
    # On renvoie le xml associé
    tree = fromDicToXml(data)
    rought = ET.tostring(tree.getroot(), 'utf-8')
    parsed = minidom.parseString(rought)
    new = parsed.toprettyxml(indent=" ")
    return new

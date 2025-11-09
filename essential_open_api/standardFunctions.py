# 
#      * Copyright (c)2008-2017 Enterprise Architecture Solutions ltd.
#      * This file is part of Essential Architecture Manager, 
#      * the Essential Architecture Meta Model and The Essential Project.
#      *
#      * Essential Architecture Manager is free software: you can redistribute it and/or modify
#      * it under the terms of the GNU General Public License as published by
#      * the Free Software Foundation, either version 3 of the License, or
#      * (at your option) any later version.
#      *
#      * Essential Architecture Manager is distributed in the hope that it will be useful,
#      * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#      * but WITHOUT ANY WARRANTY; without even the implied warranty of
#      * GNU General Public License for more details.
#      *
#      * You should have received a copy of the GNU General Public License
#      * along with Essential Architecture Manager.  If not, see <http://www.gnu.org/licenses/>.
#      * 
#       
# 19.02.2008	JWC v1.0
# 20.05.2008	JWC v1.0
# 22.05.2008	JWC v1.0 released Create or Update instances and Attributes
# 15.07.2008	JWC v1.1 Added new functions for finding instances by name in Essential.
# 17.07.2008	JWC	v1.1.1 fixed matching issue in getEssentialInstanceContainsIgnoreCase()
# 11.08.2008	JWC	v1.2 Added new functions (2) to match instance names more accurately
# 04.09.2008	JWC	v1.2.1 Updated addIfNotThere() function to ensure multiple values are not 
#						 added to single-cardinality slots. Added setSlot() function to enable this
# 20.03.2009	JWC	v1.3 Added support for importing / synchronising EA_Relation and :EA_Graph_Relation instances
# 20.03.2009	JWC	v1.3.1 Fixed the handling of the 3 name slot variants
# 20.09.2011    JWC v1.4 Handle graph relations that have no :relation_name value in the source.
# 04.10.2011    JWC v1.5 Migrated to .py, handle non-existent slots.
# 31.10.2011    JWC v1.6 Moved guard code to addIfNotThere() and setSlot() to avoid null instance.
# 28.09.2012    JWC v1.7 Changed the progress printing to remove names / external references used to support Unicode.
# 19.11.2012    JWC v1.8 Added new Instance creating and getting functions.
# 21.11.2012    JWC v1.9 Added the EssentialGetInstance() function to intelligently find the correct instance
# 12.12.2012    JWC v1.10 Added the GetActorToRole() function
# 04.04.2014    JWC v1.11 Added functions to set slots that take Class and Slot instances. Also delete / remove functions.
# 09.05.2014    JWC v1.12 Additional functions to support the delete / remove operations of the Import Utility
# 16.05.2014    JWC v1.13 Reworked EssentialDeleteInstance to filter by slots that have contents / values
# 31.05.2017    JWC v1.14 New functions to handle imports of Classes.
# 27.07.2017    JWC v2.0 Revised approach to finding instances of Classes.
# 26.02.2020    JWC v2.1 Use of the new referenced_instance slot for external references
# 27.02.2020    JWC v2.2.1 Better performance update and drop the getReferencedInstance()
# 
# Essential(tm) Architecture Manager
# Standard set of functions to be used by the integration/import scripts
#

from java.util import Date
from java.text import DateFormat
from edu.stanford.smi.protege.model import ValueType

# Get a reference to the instance of the specified class that has the specified external reference in the
# specified external repository. If such an instance cannot be found, create one with the specified 
# name (real name, not instance name)
# theClassName - the name of the Essential meta class
# theExternalRef - the unique reference/instance ID of the element in the external repository
# theExternalRepository - the name of the external repository
# theInstanceName - the name of the instance that is being created/updated in the integration
def getEssentialInstance(theClassName, theExternalRef, theExternalRepository, theInstanceName):
	anInstList = kb.getCls(theClassName).getDirectInstances()
	for anInst in anInstList:
		anExternalRefList = anInst.getDirectOwnSlotValues(kb.getSlot("external_repository_instance_reference"))
		if anExternalRefList != None:
			anExternalRef = getExternalRefInst(anExternalRefList, getExternalRepository(theExternalRepository))
			if anExternalRef != None:
				anExternalID = anExternalRef.getDirectOwnSlotValue(kb.getSlot("external_instance_reference"))
				if(anExternalID == theExternalRef):
					anExternalRef.setOwnSlotValue(kb.getSlot("external_update_date"), timestamp())
					print "Updated instance via name match, on class: " + theClassName
					return anInst
	# if we're here, we haven't found one, so create one
	aNewInst = kb.createInstance(None, kb.getCls(theClassName))
	anExternalRef = createExternalRefInst(theExternalRepository, theExternalRef)
	
	# Handle the 3 variants of name
	aNameSlot = getNameSlot(aNewInst)
	aNewInst.setOwnSlotValue(kb.getSlot(aNameSlot), theInstanceName)	
	aNewInst.addOwnSlotValue(kb.getSlot("external_repository_instance_reference"), anExternalRef)
	print "Created new instance of class: " + theClassName
	return aNewInst

# Return the external reference that applied to the specified External Repository from a list 
# of external references from an Essential instance.
# theExternalRefList - the list of external reference records for an EssentialInstance
# theExternalRepository - the instance of the external repository	
def getExternalRefInst(theExternalRefList, theExternalRepository):
    # Search the list looking for the entry about the specified External Repository
    aRef = filter(lambda x: x.getDirectOwnSlotValue(kb.getSlot("external_repository_reference")) == theExternalRepository, theExternalRefList)	
    return aRef[0]

# Create a new External Reference record to be associated with an Essential instance.
# theExternalRepository - the name (a String) of the external repository
# theExternalReference - the reference ID that is used in the specified external repository
def createExternalRefInst(theExternalRepositoryName, theExternalReference):
	aNewExternalRef = kb.createInstance(None, kb.getCls("External_Instance_Reference"))
	aNewExternalRef.setOwnSlotValue(kb.getSlot("name"), theExternalRepositoryName + "::" + theExternalReference)
	aNewExternalRef.addOwnSlotValue(kb.getSlot("external_repository_reference"), getExternalRepository(theExternalRepositoryName))
	aNewExternalRef.setOwnSlotValue(kb.getSlot("external_instance_reference"), theExternalReference)
	aNewExternalRef.setOwnSlotValue(kb.getSlot("external_update_date"), timestamp())
	return aNewExternalRef

# Get a reference to the instance of External_Repository that has the specified name
# theExternalRepositoryName - the name of the external repository
def getExternalRepository(theExternalRepositoryName):
    anExtRepos = GetInstanceOfClass("External_Repository", theExternalRepositoryName)
    if anExtRepos != None:
        return anExtRepos
        
    # report error if not found
    print "Repository: " + theExternalRepositoryName + " not defined as an external repository for integration in this EssentialAM model"
			
# Return a string of the current date/time to be used for timestamping.
def timestamp():
	return DateFormat.getDateTimeInstance().format(Date())
	
# Update the named attribute associated with the specified technology instance object
# or create it if it's not already been defined
def setOrUpdateTechInstAttributeByName(theAttributeName, theAttributeValue, theInstance):
	anAttributeList = kb.getCls("Attribute").getDirectInstances()
	aFoundAttribute = None
	for anAttribute in anAttributeList:
		if(anAttribute.getDirectOwnSlotValue(kb.getSlot("name")) == theAttributeName):
			aFoundAttribute = anAttribute
	anAttributeValList = theInstance.getDirectOwnSlotValues(kb.getSlot("technology_instance_attributes"))
	for anAttributeVal in anAttributeValList:
		if(anAttributeVal.getDirectOwnSlotValue(kb.getSlot("attribute_value_of")) == aFoundAttribute):
			anAttributeVal.setOwnSlotValue(kb.getSlot("attribute_value"), theAttributeValue)
			anAttributeVal.setOwnSlotValue(kb.getSlot("name"), theAttributeName + " = " + theAttributeValue)
			return
	# Else this instance has no attribute value defined for this attribute
	aNewAV = kb.createInstance(None, kb.getCls("Attribute_Value"))
	aNewAV.addOwnSlotValue(kb.getSlot("attribute_value_of"), aFoundAttribute)
	aNewAV.setOwnSlotValue(kb.getSlot("attribute_value"), theAttributeValue)
	aNewAV.setOwnSlotValue(kb.getSlot("name"), theAttributeName + " = " + theAttributeValue)
	theInstance.addOwnSlotValue(kb.getSlot("technology_instance_attributes"), aNewAV)
		
# Update the named attribute associated with the specified technology Node (theInstance) object
# or create it if it's not already been defined
def setOrUpdateTechNodeAttributeByName(theAttributeName, theAttributeValue, theInstance):
	anAttributeList = kb.getCls("Attribute").getDirectInstances()
	aFoundAttribute = None
	for anAttribute in anAttributeList:
		if(anAttribute.getDirectOwnSlotValue(kb.getSlot("name")) == theAttributeName):
			aFoundAttribute = anAttribute
	anAttributeValList = theInstance.getDirectOwnSlotValues(kb.getSlot("technology_node_attributes"))
	for anAttributeVal in anAttributeValList:
		if(anAttributeVal.getDirectOwnSlotValue(kb.getSlot("attribute_value_of")) == aFoundAttribute):
			anAttributeVal.setOwnSlotValue(kb.getSlot("attribute_value"), theAttributeValue)
			anAttributeVal.setOwnSlotValue(kb.getSlot("name"), theAttributeName + " = " + theAttributeValue)
			return
	# Else this instance has no attribute value defined for this attribute
	aNewAV = kb.createInstance(None, kb.getCls("Attribute_Value"))
	aNewAV.addOwnSlotValue(kb.getSlot("attribute_value_of"), aFoundAttribute)
	aNewAV.setOwnSlotValue(kb.getSlot("attribute_value"), theAttributeValue)
	aNewAV.setOwnSlotValue(kb.getSlot("name"), theAttributeName + " = " + theAttributeValue)
	theInstance.addOwnSlotValue(kb.getSlot("technology_node_attributes"), aNewAV)

# Set a slot value to the specified value. To be used with single-cardinality slots
# only
# theInstance - the instance to which we wish to set the slot to contain theInstanceToAdd
# theSlotName - the name of the slot on theInstance
# theInstanceToAdd - the instance to set in theSlotName slot on theInstance
# 04.10.2011 JWC - check that slot exists first
# 31.10.2011 JWC - Guard code, Check theInstance != None
def setSlot(theInstance, theSlotName, theInstanceToAdd):
    if theInstance == None:
        return
    aSlot = kb.getSlot(theSlotName)
    if aSlot != None:
        theInstance.setOwnSlotValue(aSlot, theInstanceToAdd)
    else:
        print "WARNING: Attempt to set non-existent slot: " + theSlotName

# Add the slot value to the specified instance only if it's not already there.
# theInstance - the instance to which we wish to add theInstanceToAdd
# theSlotName - the name of the slot on theInstance
# theInstanceToAdd - the instance to add to theSlotName slot on theInstance
# v1.2.1: If theSlotName is a single cardinality slot, use setSlot()
# 04.10.2011 JWC - check that the slot exists first
# 31.10.2011 JWC - Guard code, Check theInstance != None
def addIfNotThere(theInstance, theSlotName, theInstanceToAdd):
    if theInstance == None:
        return
    aSlot = kb.getSlot(theSlotName)
    if aSlot != None:
        if aSlot.getAllowsMultipleValues():
            anInstList = theInstance.getDirectOwnSlotValues(kb.getSlot(theSlotName))
            for anInst in anInstList:
                if anInst == theInstanceToAdd:
                    return
            # else we haven't got this already, so add it
            theInstance.addOwnSlotValue(kb.getSlot(theSlotName), theInstanceToAdd)
        else:
            setSlot(theInstance, theSlotName, theInstanceToAdd)
    else:
        print "WARNING: Attempt to set non-existent slot: " + theSlotName

# Find the instance by a contains case-sensitive match on the instance name in Essential repository
# Use this for getting instances that are expected to already be in the repository
# If not found, create a new one.
# theClassName - the Essential class for the instance
# theExternalRef - the External Reference ID for this instance
# theExternalRepository - the External Repository that theExternalRef applies to
# theInstanceName - the name of the instance in the Essential Repository
def getEssentialInstanceContains(theClassName, theExternalRef, theExternalRepository, theInstanceName):
    # 20.09.2011 JWC - handle empty theInstanceName
    if theInstanceName == "":
        theInstanceName = theExternalRef
        print "Handling empty instance name for " + theExternalRef
    
    # 20.09.2011 JWC - Make sure that the source class is in the targert repository
    aCls = kb.getCls(theClassName)    
    # if not-report a warning and skip that instance.
    if aCls == None:
        print "WARNING: Skipping instance of unknown class: " + theClassName + " : " + theInstanceName
        return None
    
    anInstList = kb.getCls(theClassName).getDirectInstances()
    for anInst in anInstList:
        anExternalRefList = anInst.getDirectOwnSlotValues(kb.getSlot("external_repository_instance_reference"))
        if anExternalRefList != None:	
            anExternalRef = getExternalRefInst(anExternalRefList, getExternalRepository(theExternalRepository))
            if anExternalRef != None:
                anExternalID = anExternalRef.getDirectOwnSlotValue(kb.getSlot("external_instance_reference"))
                if(anExternalID == theExternalRef):
                    anExternalRef.setOwnSlotValue(kb.getSlot("external_update_date"), timestamp())
                    print "Updated Instance via External Reference on class: " + theClassName
                    return anInst

	# If we're here, this external reference hasn't been found.
	# Try to find by instance_name and case sensitive
    aNameSlot = getNameSlotForClass(theClassName)
    for anInst in anInstList:
        aName = anInst.getDirectOwnSlotValue(kb.getSlot(aNameSlot))
		
        if aName != None:
            if(aName.find(theInstanceName) != -1):
                # We have found a match
                anExternalRef = createExternalRefInst(theExternalRepository, theExternalRef)
                anInst.addOwnSlotValue(kb.getSlot("external_repository_instance_reference"), anExternalRef)
                print "Updated instance via name match, on class: " + theClassName
                return anInst
				
    # if we're here, we haven't found one, so create one
    aNewInst = kb.createInstance(None, kb.getCls(theClassName))
	
	# Handle the 3 variants of name
	#aNameSlot = getNameSlot(aNewInst)
    aNewInst.setOwnSlotValue(kb.getSlot(aNameSlot), theInstanceName)	
				
    anExternalRef = createExternalRefInst(theExternalRepository, theExternalRef)
    aNewInst.addOwnSlotValue(kb.getSlot("external_repository_instance_reference"), anExternalRef)
    print "Created new instance of class: " + theClassName
    return aNewInst

# Find the instance by a contains match - ignoring case - on the instance name in Essential repository
# Use this for getting instances that are expected to already be in the repository. Use theMatchString
# to specify the string to use as a match on existing instances.
# If not found, create a new one.
# theClassName - the Essential class for the instance
# theExternalRef - the External Reference ID for this instance
# theExternalRepository - the External Repository that theExternalRef applies to
# theInstanceName - the name of the instance in the Essential Repository
# theMatchString - the string that should be used to match against and find the instance
def getEssentialInstanceContainsIgnoreCase(theClassName, theExternalRef, theExternalRepository, theInstanceName, theMatchString):
	anInstList = kb.getCls(theClassName).getDirectInstances()
	for anInst in anInstList:
		anExternalRefList = anInst.getDirectOwnSlotValues(kb.getSlot("external_repository_instance_reference"))
		if anExternalRefList != None:	
			anExternalRef = getExternalRefInst(anExternalRefList, getExternalRepository(theExternalRepository))
			if anExternalRef != None:
				anExternalID = anExternalRef.getDirectOwnSlotValue(kb.getSlot("external_instance_reference"))
				if(anExternalID == theExternalRef):
					anExternalRef.setOwnSlotValue(kb.getSlot("external_update_date"), timestamp())
					print "Updated Instance via External Reference on class: " + theClassName
					return anInst
					
	# If we're here, this external reference hasn't been found.
	# Try to find by instance_name - ignoring case of each
	aNameSlot = getNameSlotForClass(theClassName)
	for anInst in anInstList:
		aName = anInst.getDirectOwnSlotValue(kb.getSlot(aNameSlot))
		if aName != None:
			aName = aName.lower()
			aMatchName = theMatchString.lower()
			if((aName.find(aMatchName) != -1) or (aMatchName.find(aName) != -1)):
				# We have found a match
				anExternalRef = createExternalRefInst(theExternalRepository, theExternalRef)
				anInst.addOwnSlotValue(kb.getSlot("external_repository_instance_reference"), anExternalRef)
				print "Updated instance via name match, on class: " + theClassName
				return anInst
				
	# if we're here, we haven't found one, so create one
	aNewInst = kb.createInstance(None, kb.getCls(theClassName))
	aNewInst.setOwnSlotValue(kb.getSlot(aNameSlot), theInstanceName)			
	anExternalRef = createExternalRefInst(theExternalRepository, theExternalRef)
	aNewInst.addOwnSlotValue(kb.getSlot("external_repository_instance_reference"), anExternalRef)
	print "Created new instance of class: " + theClassName
	return aNewInst
	
# Define a new External Repository or ignore definition if the repository is already known
# theExternalRepository - the name of the external repository
def defineExternalRepository(theExternalRepository, theDescription):
    anInstance = GetInstanceOfClass("External_Repository", theExternalRepository)
    if anInstance != None:
        return anInstance

	# if we're here it's not defined, so define it.
    aNewRepos = kb.createInstance(None, kb.getCls("External_Repository"))
    aNewRepos.setOwnSlotValue(kb.getSlot("name"), theExternalRepository)
    aNewRepos.setOwnSlotValue(kb.getSlot("description"), theDescription)
    return aNewRepos
	
# Function to add a new Attribute instance to the Essential model.
# theName - the name of the Attribute
# theDescription - a description of the attribute
# theUnit - the units of the attribute value, e.g. MB, Mbps, kg or '_' or space if not applicable
def addNewEAMAttribute(theName, theDescription, theUnit):
	anAttributeList = kb.getCls("Attribute").getDirectInstances()
	aFoundAttribute = None
	for anAttribute in anAttributeList:
		if(anAttribute.getDirectOwnSlotValue(kb.getSlot("name")) == theName):
			aFoundAttribute = anAttribute
			break
	if aFoundAttribute == None:
		aNewAttribute = kb.createInstance(None, kb.getCls("Attribute"))
		aNewAttribute.setOwnSlotValue(kb.getSlot("name"), theName)
		aNewAttribute.setOwnSlotValue(kb.getSlot("description"), theDescription)
		aNewAttribute.setOwnSlotValue(kb.getSlot("attribute_value_unit"), theUnit)

# Function to find Technology_Node instances by a name match (precise, not contains), regardless of case
# Use this for getting instances that are expected to already be in the repository. Use theMatchString
# to specify the string to use as a match on existing instances. Matching is based on just the hostname
# and strips any trailing domain components from the name, after the first '.'
# If not found, create a new one.
# theClassName - the Essential class for the instance
# theExternalRef - the External Reference ID for this instance
# theExternalRepository - the External Repository that theExternalRef applies to
# theInstanceName - the name of the instance in the Essential Repository
# theMatchString - the string that should be used to match against and find the instance 
def getEssentialNodeInstanceIgnoreCase(theClassName, theExternalRef, theExternalRepository, theInstanceName, theMatchString):
	anInstList = kb.getCls(theClassName).getDirectInstances()
	for anInst in anInstList:
		anExternalRefList = anInst.getDirectOwnSlotValues(kb.getSlot("external_repository_instance_reference"))
		if anExternalRefList != None:	
			anExternalRef = getExternalRefInst(anExternalRefList, getExternalRepository(theExternalRepository))
			if anExternalRef != None:
				anExternalID = anExternalRef.getDirectOwnSlotValue(kb.getSlot("external_instance_reference"))
				if(anExternalID == theExternalRef):
					anExternalRef.setOwnSlotValue(kb.getSlot("external_update_date"), timestamp())
					print "Updated Instance via External Reference on class: " + theClassName
					return anInst
					
	# If we're here, this external reference hasn't been found.
	# Try to find by instance_name - ignoring case of each
	for anInst in anInstList:
		aName = anInst.getDirectOwnSlotValue(kb.getSlot("name"))
		if aName != None:
			aName = aName.split(".", 1)[0].lower()
			aMatchName = theMatchString.lower()
			if(aName == aMatchName):
				# We have found a match
				anExternalRef = createExternalRefInst(theExternalRepository, theExternalRef)
				anInst.addOwnSlotValue(kb.getSlot("external_repository_instance_reference"), anExternalRef)
				print "Updated instance via name match, on class: " + theClassName
				return anInst
				
	# if we're here, we haven't found one, so create one
	aNewInst = kb.createInstance(None, kb.getCls(theClassName))
	aNewInst.setOwnSlotValue(kb.getSlot("name"), theInstanceName)			
	anExternalRef = createExternalRefInst(theExternalRepository, theExternalRef)
	aNewInst.addOwnSlotValue(kb.getSlot("external_repository_instance_reference"), anExternalRef)
	print "Created new instance of class: " + theClassName
	return aNewInst
	
# Function to find instances by a name match (precise, not contains), regardless of case
# Use this for getting instances that are expected to already be in the repository. Use theMatchString
# to specify the string to use as a match on existing instances.
# If not found, create a new one.
# theClassName - the Essential class for the instance
# theExternalRef - the External Reference ID for this instance
# theExternalRepository - the External Repository that theExternalRef applies to
# theInstanceName - the name of the instance in the Essential Repository
# theMatchString - the string that should be used to match against and find the instance 
def getEssentialInstanceIgnoreCase(theClassName, theExternalRef, theExternalRepository, theInstanceName, theMatchString):
    anInstList = kb.getCls(theClassName).getDirectInstances()
    for anInst in anInstList:
        anExternalRefList = anInst.getDirectOwnSlotValues(kb.getSlot("external_repository_instance_reference"))
        if anExternalRefList != None:
            anExternalRef = getExternalRefInst(anExternalRefList, getExternalRepository(theExternalRepository))
            if anExternalRef != None:
                anExternalID = anExternalRef.getDirectOwnSlotValue(kb.getSlot("external_instance_reference"))
                if(anExternalID == theExternalRef):
                    anExternalRef.setOwnSlotValue(kb.getSlot("external_update_date"), timestamp())
                    print "Updated Instance via External Reference on class: " + theClassName
                    return anInst
					
    # If we're here, this external reference hasn't been found.
    # Try to find by instance_name - ignoring case of each
    aNameSlot = getNameSlotForClass(theClassName)
    for anInst in anInstList:
        aName = anInst.getDirectOwnSlotValue(kb.getSlot(aNameSlot))
        if aName != None:
            aName = aName.lower()
            aMatchName = theMatchString.lower()
            if(aName == aMatchName):
                # We have found a match
                anExternalRef = createExternalRefInst(theExternalRepository, theExternalRef)
                anInst.addOwnSlotValue(kb.getSlot("external_repository_instance_reference"), anExternalRef)
                print "Updated instance via name match, on class: " + theClassName
                return anInst
				
    # if we're here, we haven't found one, so create one
    aNewInst = kb.createInstance(None, kb.getCls(theClassName))
    aNewInst.setOwnSlotValue(kb.getSlot(aNameSlot), theInstanceName)			
    anExternalRef = createExternalRefInst(theExternalRepository, theExternalRef)
    aNewInst.addOwnSlotValue(kb.getSlot("external_repository_instance_reference"), anExternalRef)
    print "Created new instance of class: " + theClassName
    return aNewInst
	
# Find the right name slot for a given instance. If it's an EA_Class instance, the 
# name slot will be returned. If it's an EA_Relation instance, the relation_name slot
# is returned. If it's an instance of :EA_Graph_Relation, the :relation_name slot is
# returned
# theInstance the instance for which the correct name slot is required.
# returns a string name of the correct slot to use.
def getNameSlot(theInstance):
	aCorrectNameSlot = "name"
	for aSlot in theInstance.getOwnSlots():
		if(aSlot == kb.getSlot("name")):
			aCorrectNameSlot = "name"
		elif(aSlot == kb.getSlot("relation_name")):
			aCorrectNameSlot = "relation_name"
		elif(aSlot == kb.getSlot(":relation_name")):
			aCorrectNameSlot = ":relation_name"

	return aCorrectNameSlot
	
# Find the right name slot for a given class. If it's an EA_Class, the 
# name slot will be returned. If it's an EA_Relation class, the relation_name slot
# is returned. If it's an class of :EA_Graph_Relation, the :relation_name slot is
# returned
# theClassName the name of the class for which the correct name slot is required.
# returns a string name of the correct slot to use.
def getNameSlotForClass(theClassName):
	aCorrectNameSlot = "name"
	for aSlot in kb.getCls(theClassName).getTemplateSlots():
		if(aSlot == kb.getSlot("name")):
			aCorrectNameSlot = "name"
		elif(aSlot == kb.getSlot("relation_name")):
			aCorrectNameSlot = "relation_name"
		elif(aSlot == kb.getSlot(":relation_name")):
			aCorrectNameSlot = ":relation_name"

	return aCorrectNameSlot
	
#
def CreateNewEssentialInstance(theClassName, theInstanceName):
    """ Create a new instance in the repository without searching for any
        existing instances and without associating an external repository 
        reference
        
        theClassName - the name of the Class of which to create the instance
        theInstanceName - the value of the "name" slot on the new Essential instance
        
        returns a reference to the new instance
        
        19.11.2012 JWC
    """
    aNewInst = CreateNewEssentialInstanceWithID(theClassName, theInstanceName, None) 
    return aNewInst
    
#
def CreateNewEssentialInstanceWithID(theClassName, theInstanceName, theInstanceID):
    """ Create a new instance in the repository without searching for any
        existing instances and without associating an external repository 
        reference
        
        theClassName - the name of the Class of which to create the instance
        theInstanceName - the value of the "name" slot on the new Essential instance
        theInstanceID - the internal, Protege ID that should be used for this instance
        
        returns a reference to the new instance
        
        19.11.2012 JWC
    """
    # if we're here, we haven't found one, so create one
    aNewInst = kb.createInstance(theInstanceID, kb.getCls(theClassName))
	
    # Handle the 3 variants of name
    aNameSlot = getNameSlot(aNewInst)
    aNewInst.setOwnSlotValue(kb.getSlot(aNameSlot), theInstanceName)	
    print "Created new instance of class: " + theClassName
    return aNewInst

#
def GetEssentialInstanceByID(theClassName, theInstanceName, theInstanceID):
    """ Find the instance in the repository with the specified internal ID
        If not found, create one in the repository with that internal ID and with 
        the specified type and name.
        
        theClassName - the name of the Class of which to create the instance
        theInstanceName - the value of the "name" slot on the new Essential Instance
        theInstanceID - the internal, Protege ID that is being searched for and used
        for any new instance
        
        returns the new instance identified by ID
        
        19.11.2012 JWC
    """
    # Find an instance in the repository with internal ID = theInstanceID
    anInstance = kb.getInstance(theInstanceID)
    if anInstance != None:
        print "Found instance by ID on class: " + theClassName
         
    # If not found, create it
    if anInstance == None:
        anInstance = CreateNewEssentialInstanceWithID(theClassName, theInstanceName, theInstanceID)
    
    return anInstance
#

# Intelligent Essential Get Instance function.
def EssentialGetInstance(theClassName, theInstanceID, theInstanceName, theExternalID, theExternalRepositoryName):
    """ Get the Essential instance from the current repository of the specified Class. 
        Firstly, the repository will be searched for the specified instance ID (internal Protege name). 
        If no such instance can be found, the repository is searched for an instance with the specified external 
        repository instance reference. 
        If no such instance can be found, the repository is searched is for instances of the Specified class that
        has a name that exactly (case and full name) matches the specified instance name.
        If no such instance can be found, a new instance with the above the parameters is created. In this case
        Protege will automatically assign a new instance ID.
        
        theClassName - the name of the class of the instance to find. Search is scoped by class
        theInstanceID - the internal Protege name / ID for the instance. Set to "" to bypass search by instance ID
        theInstanceName - the name of the specified instance. When an instance is found by instance ID or External
                          reference, this parameter can be used to update the name (Essential name slot) of the 
                          instance.
        theExternalID - the ID that the instance has in the specified external source repository
        theExternalRepositoryName - the name of the external source repository         
        
        returns a reference to the correct Essential Instance or None if an attempt is made to create an instance
        of an unknown class.
        
        20.11.2012 JWC
    """
    anEssentialInstance = None
    
    # 20.09.2011 JWC - Make sure that the source class is in the targert repository
    aClass = kb.getCls(theClassName)    
    # if not-report a warning and skip that instance.
    if aClass == None:
        print "WARNING: Skipping instance of unknown class: " + theClassName
        return None
    
    if theInstanceID != None:
        anEssentialInstance = FindEssentialInstanceByID(theInstanceID)        
        ProcessFoundInstance(anEssentialInstance, theInstanceName, theExternalRepositoryName, theExternalID)
        if anEssentialInstance != None:
            print "Updated instance via instance ID, on class: " + theClassName
            
    # If not found, search by external repository reference
    if anEssentialInstance == None:
        # anEssentialInstance = FindEssentialInstanceByExternalRef(aClass, theExternalRepositoryName, theExternalID)
        # ProcessFoundInstance(anEssentialInstance, theInstanceName, theExternalRepositoryName, theExternalID)
        # if anEssentialInstance != None:
        #     print "Updated instance via external repository reference, on class: " + theClassName
        
        # If not found, search by name
        if anEssentialInstance == None:
            anEssentialInstance = FindEssentialInstanceByName(aClass, theInstanceName)
            UpdateOrAddExternalRef(anEssentialInstance, theExternalRepositoryName, theExternalID)
            if anEssentialInstance != None:
                print "Updated instance via name, on class: " + theClassName
        
        # If not found, create a new instance
        if anEssentialInstance == None:            
            anEssentialInstance = CreateNewEssentialInstance(theClassName, theInstanceName)
            AddExternalReferenceID(anEssentialInstance, theExternalID, theExternalRepositoryName)
    
    # Return the results of the searches.            
    return anEssentialInstance
#

def FindEssentialInstanceByID(theInstanceID):
    """ Find the instance of the specified class that has the internal ID specified.
        If not found returns None
        
        theClass - the class of the instance to find
        theInstanceID - the internal, Protege name of the instance
     
        returns a reference to the found instance or None if not found
        
        20.11.2012 JWC
    """
    anInstance = None
    if theInstanceID != None and len(theInstanceID) > 0:
        anInstance = kb.getInstance(theInstanceID)
        
    return anInstance
#

def FindEssentialInstanceByExternalRef(theClass, theExternalRepositoryName, theExternalID):
    """ Find the Essential instance of the specified class in the repository, that has
        the specified external reference.
        
        theClass - the class of the required instance
        theExternalRepositoryName - the external repository name that contains the external ID
        theExternalID - the ID of the required instance in the specified external source repository
        
        returns a reference to the found instance or None if not found
        20.11.2012 JWC
        26.02.2020 JWC updated to use the getReferencedInstance() approach if available
        27.02.2020 JWC updated again to drop the above and just use the slot. Even faster
    """
    #
    anExternalRef = None
    anInstance = None
    anExternalRepos = GetInstanceOfClass("External_Repository", theExternalRepositoryName)
    anExternalReferenceIDList = kb.getFramesWithValue(kb.getSlot("external_instance_reference"), None, 0, theExternalID)
    if anExternalReferenceIDList.size() > 0:
        anInstanceList = filter(lambda x: x.getDirectOwnSlotValue(kb.getSlot("external_repository_reference")) == anExternalRepos, anExternalReferenceIDList)
        if len(anInstanceList) > 0:
            anExternalRef = anInstanceList[0]
            
            # Got the external reference object. Now find instance to which this is associated
            # If the referenced_instance slot is available, use that to find the instance
            if kb.getSlot('referenced_instance') != None:
                anInstance = anExternalRef.getDirectOwnSlotValue(kb.getSlot('referenced_instance'))
                UpdateExternalReferenceID(anExternalRef)
            # else use the old approach
            else:
                anInstList = theClass.getDirectInstances()
                anInstanceSet = filter(lambda x: x.getDirectOwnSlotValue(kb.getSlot("external_repository_instance_reference")) == anExternalRef, anInstList)
                if len(anInstanceSet) > 0:
                    anInstance = anInstanceSet[0]
                    UpdateExternalReferenceID(anExternalRef)

    #               
    return anInstance
    
#
def FindEssentialInstanceByName(theClass, theInstanceName):
    """ Find the Essential instance of the specified class in the repository that has
        a name that exactly matches the specified name
        
        theClass - the class of the required instance
        theInstance - the name of the instance to find
        
        returns a reference to the found instance or None if not found
        
        20.11.2012 JWC
    """
    anInstance = None 
    if theInstanceName != None:
        anInstance = GetInstanceOfClass(theClass.getName(), theInstanceName.strip())
   
    return anInstance
#
def AddExternalReferenceID(theInstance, theExternalID, theExternalRepository):
    """ Add the specified external repository instance reference to the specified instance
    
        theInstance - the Protege instance to which the external reference should be created and added
        theExternalID - the ID from the external repository for the specified instance
        theExternalRepository - the external repository to which the external ID belongs
        
        20.11.2012 JWC
    """
    anExternalRef = createExternalRefInst(theExternalRepository, theExternalID)
    theInstance.addOwnSlotValue(kb.getSlot("external_repository_instance_reference"), anExternalRef)
#
#
def FindExternalReferenceID(theInstance, theExternalID, theExternalRepositoryName):
    """ Find the specified external reference for the Instance with the specified external ID on the
        specified external repository or return None
    
        theInstance - the instance to which the external ID relates
        theExternalID - the ID of this instance in the external repository
        theExternalRepositoryName - the external source repository from which the instance is being imported
    
        returns a reference the external repository object or None if not found.
        
        21.11.2012 JWC
    """
    anExternalRef = None
    anExternalRepos = getExternalRepository(theExternalRepositoryName)
    
    anExternalRefList = theInstance.getDirectOwnSlotValues(kb.getSlot("external_repository_instance_reference"))
    aRef = filter(lambda x: x.getDirectOwnSlotValue(kb.getSlot("external_repository_reference")) == anExternalRepos, anExternalRefList)
    if len(aRef) > 0:
        anExternalRef = aRef[0]
    
    return anExternalRef
            
#
def UpdateExternalReferenceID(theExternalRepositoryRef):
    """ Update the timestamp on the specified external repository reference
    
        theExternalRepositoryRef - the external repository reference to update.
        
        21.11.2012 JWC
    """
    theExternalRepositoryRef.setOwnSlotValue(kb.getSlot("external_update_date"), timestamp())
    
#
def UpdateEssentialInstanceName(theInstance, theInstanceName):
    """ Update the name of the specified instance to use the new name specified
        but only if the names do not match
        
        theInstance - the instance to be updated
        theInstanceName - the new name to use
        
        21.11.2012 JWC
    """
    if theInstance != None:
        if (theInstanceName != None) and (len(theInstanceName) > 0):
            theInstance.setOwnSlotValue(kb.getSlot("name"), theInstanceName)
#

def UpdateOrAddExternalRef(theInstance, theExternalRepositoryName, theExternalID):
    """ Find the correct External reference for the specified instance in the specified
        repository with the specified ID and update it. If no such external instance reference
        exists, create it
        
        theInstance - the instance for which the external reference should be updated or added
        theExternalRepositoryName - the name of the external repository
        theExternalID - the external ID in the external repository for the instance
        
        21.11.2012 JWC
    """
    if theInstance != None:
        anExternalReposRef = FindExternalReferenceID(theInstance, theExternalID, theExternalRepositoryName)
        if anExternalReposRef != None:            
            UpdateExternalReferenceID(anExternalReposRef)
        else:
            AddExternalReferenceID(theInstance, theExternalID, theExternalRepositoryName)
#

def ProcessFoundInstance(theInstance, theInstanceName, theExternalRepositoryName, theExternalID):
    """ Process an Essential instance that has been found by updating the relevant external repository instance
        reference. If theInstanceName contains a non-empty value, change the Essential name slot value
        for theInstance
        
        theInstance - the Essential instance to process
        theInstanceName - the new/updated name for the instance. Sets the Essential Meta Model name slot
        theExternalRepositoryName - the name of the external source repository
        theExternalID - the ID that the specified instance has in the external source repository
        
        21.11.2012 JWC       
    """
    if theInstance != None:
        # Update the external reference instance for the instance
        UpdateOrAddExternalRef(theInstance, theExternalRepositoryName, theExternalID)
        
        # If theInstanceName is populated (not None or "") change the name of theInstance
        if (theInstanceName != None) and (len(theInstanceName) > 0):
            UpdateEssentialInstanceName(theInstance, theInstanceName)
#
# Get an existing or new Actor To Role relation for the specified Actor and Role
def GetActorToRole(theActor, theRole, theExtRepos):
    """ Get an existing or new Actor To Role relation for the specified Actor and Role
    
        theActor - the Actor instance to add to the role
        theRole - the Role that the Actor is to play
        theExtRepos - the name of the External repository from which the actors and roles are being
                      imported.
    
        returns the relevant ActorToRole relation or None if Actor or Role is invalid
    """
    anActToRole = None
    if (theActor != None) and (theRole != None):
        anActorName = theActor.getOwnSlotValue(kb.getSlot("name"))
        aRoleName = theRole.getOwnSlotValue(kb.getSlot("name"))
        anActToRoleName = anActorName + " as " + aRoleName
        anActToRole = EssentialGetInstance("ACTOR_TO_ROLE_RELATION", "", anActToRoleName, anActToRoleName, theExtRepos)
        addIfNotThere(anActToRole, "act_to_role_from_actor", theActor)        
        addIfNotThere(anActToRole, "act_to_role_to_role", theRole)
            
    return anActToRole
#

# Functions to add Class instances and Slot instances to Slots on an instance
def addClassToSlot(theInstance, theSlotName, theClassName):
    """ Add a Class instance to a Slot on the specified instance, where the Slot takes a Class type.
    
        theInstance - the instance on which to set the slot
        theSlotName - the slot to which the Class should be added
        theClassName - the name of the Class instance to add to the slot
    """
    if theInstance == None:
        return
    
    aCls = kb.getCls(theClassName)
    addIfNotThere(theInstance, theSlotName, aCls)
    
def addSlotToSlot(theInstance, theSlotName, theSlotInstanceName):
    """ Add a Slot instance to a Slot on the specified, where the Slot takes a Slot type
    
        theInstance - the instance on which to set the slot
        theSlotName - the slot to which the Class should be added
        theSlotInstanceName - the name of the Slot instance to add to the slot
    """
    if theInstance == None:
        return
        
    aSlot = kb.getSlot(theSlotInstanceName)
    addIfNotThere(theInstance, theSlotName, aSlot)


# Find the specified instance in the specified slot of theRelatedInstnace and remove if there.
def findAndRemoveInstanceFromSlot(theInstanceToRemove, theRelatedInstance, theSlot):
    if (theRelatedInstance != None) and (theSlot != None):
        anInstList = theRelatedInstance.getOwnSlotValues(theSlot)
        if anInstList != None:
            if anInstList.contains(theInstanceToRemove):
                theRelatedInstance.removeOwnSlotValue(theSlot, theInstanceToRemove)
                
# Remove specified instance from all architecture states
def removeInstanceFromStates(theInstance):
    # Do this from the state end of things - a bit brute force but probably safer
    aStateList = kb.getCls("Architecture_State").getDirectInstances()
    for aState in aStateList:
        # Look for the instance in each state slot and remove if it's there        
        findAndRemoveInstanceFromSlot(theInstance, aState, kb.getSlot("arch_state_application_conceptual"))
        findAndRemoveInstanceFromSlot(theInstance, aState, kb.getSlot("arch_state_application_logical"))
        findAndRemoveInstanceFromSlot(theInstance, aState, kb.getSlot("arch_state_application_physical"))
        findAndRemoveInstanceFromSlot(theInstance, aState, kb.getSlot("arch_state_application_relations"))
        findAndRemoveInstanceFromSlot(theInstance, aState, kb.getSlot("arch_state_business_conceptual"))
        findAndRemoveInstanceFromSlot(theInstance, aState, kb.getSlot("arch_state_business_logical"))
        findAndRemoveInstanceFromSlot(theInstance, aState, kb.getSlot("arch_state_business_physical"))
        findAndRemoveInstanceFromSlot(theInstance, aState, kb.getSlot("arch_state_business_relations"))
        findAndRemoveInstanceFromSlot(theInstance, aState, kb.getSlot("arch_state_information_conceptual"))
        findAndRemoveInstanceFromSlot(theInstance, aState, kb.getSlot("arch_state_information_logical"))
        findAndRemoveInstanceFromSlot(theInstance, aState, kb.getSlot("arch_state_physical_information"))
        findAndRemoveInstanceFromSlot(theInstance, aState, kb.getSlot("arch_state_security_management"))
        findAndRemoveInstanceFromSlot(theInstance, aState, kb.getSlot("arch_state_technology_conceptual"))
        findAndRemoveInstanceFromSlot(theInstance, aState, kb.getSlot("arch_state_technology_logical"))
        findAndRemoveInstanceFromSlot(theInstance, aState, kb.getSlot("arch_state_technology_physical"))
        findAndRemoveInstanceFromSlot(theInstance, aState, kb.getSlot("arch_state_technology_relations"))        

        

def deleteInstance(theInstance):
    """ Delete the specified instance from the KB
    
        theInstance - the instance to delete
    """
    if theInstance != None:
        # Remove from Taxonomies
        aSlot = kb.getSlot("element_classified_by")
        aClassifiedList = theInstance.getOwnSlotValues(aSlot)
        for aTax in aClassifiedList:
            theInstance.removeOwnSlotValue(aSlot, aTax)
            
        # Remove from Architecture States
        removeInstanceFromStates(theInstance)
        
        # Remove any associated external references
        aRefSlot = kb.getSlot("external_repository_instance_reference")
        aRefList = theInstance.getOwnSlotValues(aRefSlot)
        for aRef in aRefList:
            theInstance.removeOwnSlotValue(aRefSlot, aRef)
            kb.deleteInstance(aRef)
            
        kb.deleteInstance(theInstance)

def removeInstanceFromSlot(theInstance, theSlotName, theInstanceToRemove):
    """ Remove the specified instance from the slot on theInstance
    
        theInstance - the instance to update, on which theInstanceToRemove should be removed from theSlot
        theSlotName - the name of the Slot from which theInstanceToRemove should be removed
        theInstanceToRemove - the instance that should be removed from theSlotName, breaking a relationship
        
    """
    aSlot = kb.getSlot(theSlotName)
    if(aSlot != None):
        findAndRemoveInstanceFromSlot(theInstanceToRemove, theInstance, aSlot)
    
def clearPrimitiveSlotValue(theInstance, theSlotName):
    """ Clear the value of the specified primitive Slot, i.e. a String, Integer, Float, Boolean, Symbol slot
    
        theInstance - the instance on which to clear the specified slot
        theSlotName - the name of the slot to clear
    """
    
    if (theInstance != None):
        aSlot = kb.getSlot(theSlotName)
        
        if theInstance.getOwnSlotValueType(aSlot) == ValueType.FLOAT:
            theInstance.setOwnSlotValue(aSlot, None)
        else:
            aCurrentObject = theInstance.getDirectOwnSlotValue(aSlot)
            if(aCurrentObject != None):
                theInstance.removeOwnSlotValue(aSlot, aCurrentObject)


def deleteInstanceIfEmptySlots(theInstance, theSlotList):
    """
        Delete the specified instance but only if there are no values in the slots
        defined in theSlotList
        
        theInstance - the instance to delete
        theSlotList - a list of slot names to check for empty values
        
    """
    isValueInSlot = False
    
    if(theInstance != None):
        if(theSlotList != None):
    
            for aSlot in theSlotList:
                if (aSlot != None) and (len(aSlot) >  0):
                    aValue = theInstance.getDirectOwnSlotValue(kb.getSlot(aSlot))    
                    if(aValue != None):
                        isValueInSlot = True
                        break
        
        if(not(isValueInSlot)):
            deleteInstance(theInstance)
                
                    
                

def EssentialDeleteInstance(theClassName, theInstanceID, theInstanceName, theExternalID, theExternalRepositoryName, theSlotList=None):
    """ Get and then delete the Essential instance from the current repository of the specified Class. 
        Firstly, the repository will be searched for the specified instance ID (internal Protege name). 
        If no such instance can be found, the repository is searched for an instance with the specified external 
        repository instance reference. 
        If no such instance can be found, the repository is searched is for instances of the Specified class that
        has a name that exactly (case and full name) matches the specified instance name.
        If no such instance can be found, no action will be taken.
        
        theClassName - the name of the class of the instance to find. Search is scoped by class
        theInstanceID - the internal Protege name / ID for the instance. Set to "" to bypass search by instance ID
        theInstanceName - the name of the specified instance. When an instance is found by instance ID or External
                          reference, this parameter can be used to update the name (Essential name slot) of the 
                          instance.
        theExternalID - the ID that the instance has in the specified external source repository
        theExternalRepositoryName - the name of the external source repository
        theSlotList - a list of names that defines a set slots that must be tested for any values on the instance
                        If any values are found in any of the slots in the list on the instance, the instance will NOT 
                        be deleted. Empty or None list results in delete of instance whether it has values or not in its Slots        
                
        09.05.2014 JWC
        16.05.2014 JWC Added theSlotList
    """
    anEssentialInstance = None
    
    # 20.09.2011 JWC - Make sure that the source class is in the targert repository
    aClass = kb.getCls(theClassName)    
    # if not-report a warning and skip that instance.
    if aClass == None:
        print "WARNING: Skipping instance of unknown class: " + theClassName
        return
    
    if theInstanceID != None:
        anEssentialInstance = FindEssentialInstanceByID(theInstanceID)        
        if anEssentialInstance != None:
            deleteInstanceIfEmptySlots(anEssentialInstance, theSlotList)
            print "Deleted instance via instance ID, on class: " + theClassName
            return
            
    # If not found, search by external repository reference
    if anEssentialInstance == None:
        anEssentialInstance = FindEssentialInstanceByExternalRef(aClass, theExternalRepositoryName, theExternalID)
        if anEssentialInstance != None:
            deleteInstanceIfEmptySlots(anEssentialInstance, theSlotList)
            print "Delete instance via external repository reference, on class: " + theClassName
            return
        
        # If not found, search by name
        if anEssentialInstance == None:
            anEssentialInstance = FindEssentialInstanceByName(aClass, theInstanceName)            
            if anEssentialInstance != None:
                deleteInstanceIfEmptySlots(anEssentialInstance, theSlotList)
                print "Deleted instance via name, on class: " + theClassName
                return
        
        # If not found, create a new instance
        if anEssentialInstance == None:            
            print "Delete instance failed: Unable to find specified instance of class: " + theClassName
    
    return

# Remove and Delete an instance from a slot
def deleteInstanceFromSlot(theInstance, theSlotName, theInstanceToDelete):
    """
        Remove and delete the specified instance from the named slot on a selected instance
        
        theInstance - the instance from which the specified instance is to be deleted
        theSlotName - the name of the slot on theInstance from which theInstanceToDelete is to be deleted
        theInstanceToDelete - the instance that is to be deleted
        
    """
    # Check that theInstanceToDelete is there in the specified slot 
    theSlot = kb.getSlot(theSlotName)
   
    if (theInstance != None) and (theSlot != None):
        anInstList = theInstance.getOwnSlotValues(theSlot)
        if anInstList != None:
            if anInstList.contains(theInstanceToDelete):
                deleteInstance(theInstanceToDelete)
                print "Deleted instance from slot: " + theSlotName

# Remove all of the values from a slot on an instnace
def removeAllValuesFromSlot(theInstance, theSlotName):
    """
        Remove all the values from the slot defined by theSlotName on the specified instance.
        Slot values can be primitives or instances
        
        theInstance - the instance from which to clear the specified slot
        theSlotName - the name of the slot from which to clear all the values
    """
    aSlot = kb.getSlot(theSlotName)
    if (theInstance != None) and (aSlot != None):
        aSlotType = theInstance.getOwnSlotValueType(aSlot)
        if aSlotType == ValueType.INSTANCE:
            aValueList = theInstance.getOwnSlotValues(aSlot)
            if aValueList != None:
                for aVal in aValueList:
                    removeInstanceFromSlot(theInstance, theSlotName, aVal)
        else:
            clearPrimitiveSlotValue(theInstance, theSlotName)


# Delete all of the values from a slot on an instance
def deleteAllValuesFromSlot(theInstance, theSlotName):
    """ 
        Remove and delete all the values from the slot defined by theSlotName on the specified instance.
        Slot values can be primitives - in which case the values are cleared - or instances that are then deleted
        
        theInstance - the instance from which to clear the specified slot
        theSlotName - the name of the slot from which to delete all of the values
    """
    aSlot = kb.getSlot(theSlotName)
    if (theInstance != None) and (aSlot != None):
        aSlotType = theInstance.getOwnSlotValueType(aSlot)
        if aSlotType == ValueType.INSTANCE:        
            aValueList = theInstance.getOwnSlotValues(aSlot)
            if aValueList != None:
                for aVal in aValueList:                    
                    deleteInstanceFromSlot(theInstance, theSlotName, aVal)
        else:
            clearPrimitiveSlotValue(theInstance, theSlotName)

# GetNameSlot()
def GetNameSlot(theClassName):
    """ Get the slot that the named class uses for the name.
        for :META-CLASS things this is :NAME
        for EA_Class this is name
        for EA_Relation this is relation_name
        for :EA_Graph_Relation this is :relation_name
    
        theClassName: the name of the class to get the name slot for
        returns the Slot object for the correct name slot
    """
    aClass = kb.getCls(theClassName)
    aNameSlot = kb.getSlot("name")
    if aClass != None:
        if aClass.hasSuperclass(kb.getCls("EA_Relation")):
            aNameSlot = kb.getSlot("relation_name")
        elif aClass.hasSuperclass(kb.getCls(":EA_Graph_Relation")):
            aNameSlot = kb.getSlot(":relation_name")
        elif aClass.hasSuperclass(kb.getCls(":META-CLASS")):
            aNameSlot = kb.getSlot(":NAME")

    return aNameSlot

# getInstanceByName(theClassName, theInstanceName)
def getInstanceByName(theClassName, theInstanceName):
    """
        Look for an instance of theClassName with name slot = theInstanceName.
        If found, return that instance, otherwise create it.
        
        theClassName - the name of the Class of which we are searching for an instance
        theInstanceName - the name of the Instance that we are searching for.
    """    
    anInst = GetInstanceOfClass(theClassName, theInstanceName)
    if anInst != None:
        return anInst
        
    # otherwise, an instance of the class with name=theInstanceName has not been found
    # Handle attempts to create instances of :STANDARD-CLASS or :STANDARD-SLOT
    aClass = kb.getCls(theClassName)
    if aClass == None:
        return None
    if aClass.hasSuperclass(kb.getCls(":META-CLASS")):
        print "*** Your meta model is incompatible with this script. Attempt to create instance of Class or Slot that is not in your target repository."
        print "------ Meta Class: " + theClassName + " Instance: " + theInstanceName 
        if(theClassName == ":STANDARD-CLASS"):
            aNewInst = GetInstanceByName(theClassName, "EA_Class")
            return aNewInst
        elif(theClassName == ":STANDARD-SLOT"):
            aNewInst = GetInstanceByName(theClassName, "name")
            return aNewInst
        else:
            return None
	    # Finished handling instances of classes and slots
    aNewInst = kb.createInstance(None, kb.getCls(theClassName))
    aNewInst.setDirectOwnSlotValue(aNameSlot, theInstanceName)
    return aNewInst

# GetInstanceOfClass(theClassName, theInstanceName)
def GetInstanceOfClass(theClassName, theInstanceName):
    """
        Newest implementation of instance finding function. Designed to perform
        better en masse, using the Java API and Python filter() function.
        Find the instance of the specified class that has it's name = theInstanceName
        If found, return that Instance, otherwise return None
        
        theClassName - the name of the Class of which we are searching for an instance
        theInstanceName - the name of the instance that we are searching for
    """    
    anInstance = None
    aNameSlot = GetNameSlot(theClassName)
    anInstanceList = kb.getFramesWithValue(aNameSlot, None, 0, theInstanceName)
    if anInstanceList.size() > 0:
        anInstancesOfClass = filter(lambda x: x.hasDirectType(kb.getCls(theClassName)), anInstanceList)
        if len(anInstancesOfClass) > 0:
            anInstance = anInstancesOfClass[0]
    return anInstance
#

import json
import warnings
from io import StringIO
from typing import List

import numpy as np
import slicer
import vtk
from slicer.ScriptedLoadableModule import ScriptedLoadableModuleLogic
from .mer_support import Point, EntryTarget, ElectrodeRecord


class Feature():

    def __init__(self, projectTo):
        self.projectTo = projectTo

    def addSourceNode(self, sourceNodeID, property, visible):
        shNode = slicer.mrmlScene.GetSubjectHierarchyNode()
        id = shNode.GetItemByDataNode(slicer.util.getNode(sourceNodeID))
        shNode.SetItemAttribute(id, 'LeadORFeature', self.projectTo)
        shNode.SetItemAttribute(id, 'Property', property)
        shNode.SetItemAttribute(id, 'Visible', str(int(bool(visible))))

    def update(self):
        sourceNodes, channelNames = self.getSourceNodesProjectingToChannels()
        for channelName in channelNames:
            trajectory = Trajectory.GetTrajectoryFromChannelName(channelName)
            if trajectory is None:
                continue
            mergedData = self.getSourceNodesMergedDataForChannel(channelName, sourceNodes)
            if mergedData is None:
                slicer.util.getNode(trajectory.featuresTubeModelNodeID).GetDisplayNode().SetVisibility(0)
            else:
                vtkArrays = {key: self.getNormalizedVTKArrayWithName(mergedData[key], key) for key in
                             ['Radius', 'Color']}
                # Map feature
                if self.projectTo == 'Tube':
                    trajectory.updateTubeModelFromValues(mergedData['RecordingSiteDTT'], vtkArrays['Radius'],
                                                         vtkArrays['Color'])
                    slicer.util.getNode(trajectory.featuresTubeModelNodeID).GetDisplayNode().SetVisibility(1)
                if self.projectTo == 'Markups':
                    trajectory.updateMarkupsFromValues(mergedData['RecordingSiteDTT'], vtkArrays['Radius'],
                                                       vtkArrays['Color'])
                    slicer.util.getNode(trajectory.featuresMarkupsNodeID).GetDisplayNode().SetVisibility(1)

    def getSourceNodesMergedDataForChannel(self, channelName, sourceNodes):
        try:
            import pandas as pd
        except:
            slicer.util.pip_install('pandas')
            import pandas as pd
        shNode = slicer.mrmlScene.GetSubjectHierarchyNode()
        mergedData = None
        for sourceNode in sourceNodes:
            id = shNode.GetItemByDataNode(sourceNode)
            df = pd.read_csv(StringIO(sourceNode.GetText()))
            if (channelName not in df.columns) or (not bool(int(shNode.GetItemAttribute(id, 'Visible')))):
                continue
            df = df[['RecordingSiteDTT', channelName]]
            df = df.rename({channelName: shNode.GetItemAttribute(id, 'Property')}, axis='columns')
            if mergedData is None:
                mergedData = df
            else:
                mergedData = pd.merge(mergedData, df, on='RecordingSiteDTT')
        if (mergedData is None) or (
                not pd.Series(['RadiusAndColor', 'Radius', 'Color']).isin(mergedData.columns).any()):
            return
        if 'RadiusAndColor' in mergedData.columns:
            mergedData['Radius'] = mergedData.loc[:, 'RadiusAndColor']
            mergedData['Color'] = mergedData.loc[:, 'RadiusAndColor']
        if 'Radius' not in mergedData.columns:
            mergedData['Radius'] = pd.Series(np.ones(mergedData.size))
        if 'Color' not in mergedData.columns:
            mergedData['Color'] = pd.Series(np.ones(mergedData.size))
        return mergedData

    def getNormalizedVTKArrayWithName(self, npArray, name):
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', r'All-NaN (slice|axis) encountered')
            valuesMedian = np.nanmedian(npArray[:min(len(npArray), 5)])
        vtkValuesArray = vtk.vtkDoubleArray()
        for value in npArray:
            rel_val_from_cero = np.nanmax([(value / valuesMedian) - 1, 0.1])
            rel_val_from_cero_to_one = np.min(
                [rel_val_from_cero / 2.0, 1])  # values greater than three times the median are caped to one
            vtkValuesArray.InsertNextTuple((rel_val_from_cero_to_one,))
        vtkValuesArray.SetName(name)
        return vtkValuesArray

    def getSourceNodesProjectingToChannels(self):
        sourceNodes = []
        channelNames = []
        shNode = slicer.mrmlScene.GetSubjectHierarchyNode()
        vtk_ids = vtk.vtkIdList()
        shNode.GetItemChildren(shNode.GetSceneItemID(), vtk_ids, True)
        IDs = [vtk_ids.GetId(i) for i in range(vtk_ids.GetNumberOfIds())]
        for ID in IDs:
            if 'LeadORFeature' in shNode.GetItemAttributeNames(ID):
                if shNode.GetItemAttribute(ID, 'LeadORFeature') == self.projectTo:
                    sourceNode = shNode.GetItemDataNode(ID)
                    sourceNodes.append(sourceNode)
                    for ch in sourceNode.GetText().splitlines()[0].split(",")[1:]:
                        channelNames.append(ch)
        return sourceNodes, set(channelNames)


class Trajectory():

    def __init__(self, entry_target: EntryTarget, el_records: List[ElectrodeRecord] ):
        print(entry_target)
        shNode = slicer.mrmlScene.GetSubjectHierarchyNode()

        self.trajectoryNumber = 0
        self.channelName = ''
        self.folderID = shNode.CreateFolderItem(shNode.GetSceneItemID(),
                                                "LeadOR Trajectory " + str(self.trajectoryNumber))
        shNode.SetItemAttribute(self.folderID, 'LeadORTrajectory', str(self.trajectoryNumber))
        shNode.SetItemAttribute(self.folderID, 'ChannelName', self.channelName)
        transformID = self.createTranslationTransform()
        self.createMEModel(entry_target,el_records=el_records)
        self.createTrajectoryLine(transformID, e_t=entry_target)
        self.createTipFiducial(transformID)
        self.createFeaturesTubeModel()
        self.createFeaturesMarkups()
        shNode.SetItemExpanded(self.folderID, False)

        self.translationTransformNodeID = shNode.GetItemAttribute(self.folderID, 'translationTransformNodeID')
        self.microElectrodeModelNodeID = shNode.GetItemAttribute(self.folderID, 'microElectrodeModelNodeID')
        self.trajectoryLineNodeID = shNode.GetItemAttribute(self.folderID, 'trajectoryLineNodeID')
        self.tipFiducialNodeID = shNode.GetItemAttribute(self.folderID, 'tipFiducialNodeID')
        self.featuresTubeModelNodeID = shNode.GetItemAttribute(self.folderID, 'featuresTubeModelNodeID')
        self.featuresMarkupsNodeID = shNode.GetItemAttribute(self.folderID, 'featuresMarkupsNodeID')

        self.setNodeNames()

    def setModelVisibility(self, visible):
        slicer.util.getNode(self.microElectrodeModelNodeID).GetDisplayNode().SetVisibility3D(visible)

    def setLineVisibility(self, visible):
        slicer.util.getNode(self.trajectoryLineNodeID).GetDisplayNode().SetVisibility3D(visible)

    def setTipVisibility(self, visible):
        slicer.util.getNode(self.tipFiducialNodeID).GetDisplayNode().SetVisibility3D(visible)

    def setDistanceToTargetTransformID(self, distanceToTargetTransformID):
        planningTransformID = slicer.util.getNode(distanceToTargetTransformID).GetTransformNodeID()
        slicer.util.getNode(self.translationTransformNodeID).SetAndObserveTransformNodeID(distanceToTargetTransformID)
        slicer.util.getNode(self.featuresTubeModelNodeID).SetAndObserveTransformNodeID(planningTransformID)

    def setChannelName(self, channelName):
        self.channelName = channelName
        slicer.mrmlScene.GetSubjectHierarchyNode().SetItemAttribute(self.folderID, 'ChannelName', self.channelName)
        self.setNodeNames()

    def setNodeNames(self):
        name = self.channelName if self.channelName != '' else str(self.trajectoryNumber)
        shNode = slicer.mrmlScene.GetSubjectHierarchyNode()
        shNode.SetItemName(self.folderID, "LeadOR Trajectory %s" % name)
        slicer.util.getNode(self.translationTransformNodeID).SetName("%s: Translation Transform" % name)
        slicer.util.getNode(self.microElectrodeModelNodeID).SetName("%s: ME Model" % name)
        slicer.util.getNode(self.trajectoryLineNodeID).SetName("%s: Trajectory Line" % name)
        slicer.util.getNode(self.tipFiducialNodeID).SetName("%s: Tip Fiducial" % name)
        slicer.util.getNode(self.featuresTubeModelNodeID).SetName("%s: Feature Tube Model" % name)
        slicer.util.getNode(self.featuresMarkupsNodeID).SetName("%s: Feature Markups" % name)

    def addNodeAndAttributeToSHFolder(self, node, attributeName):
        shNode = slicer.mrmlScene.GetSubjectHierarchyNode()
        shNode.SetItemParent(shNode.GetItemByDataNode(node), self.folderID)
        shNode.SetItemAttribute(self.folderID, attributeName, node.GetID())

    def createTranslationTransform(self):
        index = np.array(np.unravel_index(self.trajectoryNumber, (3, 3))) - 1  # from 0:8 to (-1,-1) : (1,1)
        # get translation ammount
        np.seterr(divide='ignore')
        hypotenuse = 2
        alpha = np.arctan(abs(np.divide(index[0], index[1]))) if any(index) else 0
        # create matrix
        m = vtk.vtkMatrix4x4()
        m.SetElement(0, 3, -index[0] * np.sin(alpha) * hypotenuse)
        m.SetElement(1, 3, -index[1] * np.cos(alpha) * hypotenuse)
        # add node
        transformNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLLinearTransformNode')
        transformNode.SetMatrixTransformToParent(m)
        self.addNodeAndAttributeToSHFolder(transformNode, 'translationTransformNodeID')
        return transformNode.GetID()

    def createMEModel(self, entryTarget, el_records: List[ElectrodeRecord]):
        # Create a vtkPoints object to store the points
        points = vtk.vtkPoints()
        for el_rec in el_records:
            points.InsertNextPoint(el_rec.location.x, el_rec.location.y, el_rec.location.z)

        # Create a cell array to store the lines in and add the lines to it
        lines = vtk.vtkCellArray()
        lines.InsertNextCell(len(el_records))
        for i in range(len(el_records)):
            lines.InsertCellPoint(i)

        # create tube radius array


        tubeRadius = vtk.vtkDoubleArray()
        tubeRadius.SetNumberOfTuples(len(el_records))
        tubeRadius.SetName("TubeRadius")
        for i in range(len(el_records)):
            tubeRadius.SetTuple1(i, 0.3* el_records[i].record)

        linePolyData = vtk.vtkPolyData()
        linePolyData.SetPoints(points)
        linePolyData.SetLines(lines)
        linePolyData.GetPointData().AddArray(tubeRadius)
        linePolyData.GetPointData().SetActiveScalars("TubeRadius")
        tubeSegmentFilter = vtk.vtkTubeFilter()
        tubeSegmentFilter.SetInputData(linePolyData)

        tubeSegmentFilter.SetNumberOfSides(16)  # INSERT VALUE
        tubeSegmentFilter.CappingOn()
        tubeSegmentFilter.SetVaryRadiusToVaryRadiusByAbsoluteScalar()
        tubeSegmentFilter.Update()

        input1 = vtk.vtkPolyData()
        input2 = vtk.vtkPolyData()
        input1.ShallowCopy(tubeSegmentFilter.GetOutput())
        # input2.ShallowCopy(cone.GetOutput())
        appendFilter = vtk.vtkAppendPolyData()
        appendFilter.AddInputData(input1)
        appendFilter.AddInputData(input2)
        appendFilter.Update()
        cleanFilter = vtk.vtkCleanPolyData()
        cleanFilter.SetInputConnection(appendFilter.GetOutputPort())
        cleanFilter.Update()
        # add model node
        modelsLogic = slicer.modules.models.logic()
        model = modelsLogic.AddModel(cleanFilter.GetOutput())
        model.CreateDefaultSequenceDisplayNodes()
        model.CreateDefaultDisplayNodes()
        model.GetDisplayNode().SetColor(0, 1, 1)
        model.GetDisplayNode().SetVisibility2D(1)
        model.GetDisplayNode().SetVisibility3D(1)
        self.addNodeAndAttributeToSHFolder(model, 'microElectrodeModelNodeID')

        # self.setDefaultOrientationToModel(model)
        # model.SetAndObserveTransformNodeID(parentTransformID)

    # def createMEModel(self, entryTarget:  EntryTarget):
    #     # create cylinder (macro)
    #     # Create a line
    #     lineSource = vtk.vtkLineSource()
    #     lineSource.SetPoint1(entryTarget.entry.x, entryTarget.entry.y, entryTarget.entry.z)
    #     lineSource.SetPoint2(entryTarget.target.x, entryTarget.target.y, entryTarget.target.z)
    #     lineSource.Update()
    #
    #
    #     cylinder = vtk.vtkTubeFilter()
    #     cylinder.SetRadius(0.35)
    #     cylinder.SetInputData(lineSource.GetOutput())
    #
    #
    #
    #
    #     cylinder.Update()
    #     # cone line (micro)
    #     # cone = vtk.vtkConeSource()
    #     # cone.SetRadius(0.2)
    #     # cone.SetHeight(3)
    #     # cone.SetDirection(0, -1, 0)
    #     # cone.SetCenter(0, 1.5, 0)
    #     # cone.SetResolution(12)
    #     # cone.Update()
    #     # append cone and cylinder
    #     input1 = vtk.vtkPolyData()
    #     input2 = vtk.vtkPolyData()
    #     input1.ShallowCopy(cylinder.GetOutput())
    #     #input2.ShallowCopy(cone.GetOutput())
    #     appendFilter = vtk.vtkAppendPolyData()
    #     appendFilter.AddInputData(input1)
    #     appendFilter.AddInputData(input2)
    #     appendFilter.Update()
    #     cleanFilter = vtk.vtkCleanPolyData()
    #     cleanFilter.SetInputConnection(appendFilter.GetOutputPort())
    #     cleanFilter.Update()
    #     # add model node
    #     modelsLogic = slicer.modules.models.logic()
    #     model = modelsLogic.AddModel(cleanFilter.GetOutput())
    #     model.CreateDefaultSequenceDisplayNodes()
    #     model.CreateDefaultDisplayNodes()
    #     model.GetDisplayNode().SetColor(0, 1, 1)
    #     model.GetDisplayNode().SetVisibility2D(1)
    #     model.GetDisplayNode().SetVisibility3D(1)
    #     #self.setDefaultOrientationToModel(model)
    #     #model.SetAndObserveTransformNodeID(parentTransformID)
    #     self.addNodeAndAttributeToSHFolder(model, 'microElectrodeModelNodeID')

    def createTrajectoryLine(self, parentTransformID, e_t: EntryTarget):
        ls = vtk.vtkLineSource()
        ls.SetPoint1(e_t.entry.x, e_t.entry.y, e_t.entry.z)
        ls.SetPoint2(e_t.target.x, e_t.target.y, e_t.target.z)
        ls.Update()
        model = slicer.modules.models.logic().AddModel(ls.GetOutput())
        model.CreateDefaultSequenceDisplayNodes()
        model.CreateDefaultDisplayNodes()
        model.SetDisplayVisibility(1)
        model.GetDisplayNode().SetColor(0.9, 0.9, 0.9)
        self.setDefaultOrientationToModel(model)
        # model.SetAndObserveTransformNodeID(parentTransformID)
        self.addNodeAndAttributeToSHFolder(model, 'trajectoryLineNodeID')

    def createTipFiducial(self, parentTransformID):
        fiducialNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsFiducialNode')
        fiducialNode.GetDisplayNode().SetTextScale(0)
        fiducialNode.GetDisplayNode().SetVisibility3D(1)
        fiducialNode.AddControlPointWorld(vtk.vtkVector3d([0, 0, 0]))
        fiducialNode.SetLocked(True)
        fiducialNode.SetAndObserveTransformNodeID(parentTransformID)
        self.addNodeAndAttributeToSHFolder(fiducialNode, 'tipFiducialNodeID')

    def setDefaultOrientationToModel(self, model):
        # put in I-S axis
        vtkTransform = vtk.vtkTransform()
        vtkTransform.RotateWXYZ(90, 1, 0, 0)
        model.ApplyTransform(vtkTransform)

    def createFeaturesTubeModel(self):
        tubeModel = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLModelNode')
        tubeModel.CreateDefaultDisplayNodes()
        tubeModel.GetDisplayNode().SetAndObserveColorNodeID(slicer.util.getNode('Viridis').GetID())
        tubeModel.GetDisplayNode().ScalarVisibilityOn()
        self.addNodeAndAttributeToSHFolder(tubeModel, 'featuresTubeModelNodeID')

    def createFeaturesMarkups(self):
        featuresMarkups = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsFiducialNode')
        self.addNodeAndAttributeToSHFolder(featuresMarkups, 'featuresMarkupsNodeID')

    def updateTubeModelFromValues(self, samplePoints, tubeRadiusValues, tubeColorValues):
        matrix = vtk.vtkMatrix4x4()
        slicer.util.getNode(self.translationTransformNodeID).GetMatrixTransformToParent(matrix)
        transformedPoint = np.zeros(4)
        samplePointsVTK = vtk.vtkPoints()
        for i in range(samplePoints.shape[0]):
            matrix.MultiplyPoint(np.array([0.0, 0.0, samplePoints[i], 1.0]), transformedPoint)
            samplePointsVTK.InsertNextPoint(transformedPoint[:-1])
        # line source
        polyLineSource = vtk.vtkPolyLineSource()
        polyLineSource.SetPoints(samplePointsVTK)
        polyLineSource.Update()
        # poly data
        polyData = polyLineSource.GetOutput()
        polyData.GetPointData().AddArray(tubeRadiusValues)
        polyData.GetPointData().AddArray(tubeColorValues)
        polyData.GetPointData().SetScalars(tubeRadiusValues)
        # run tube filter
        tubeFilter = vtk.vtkTubeFilter()
        tubeFilter.SetInputData(polyData)
        tubeFilter.SetVaryRadiusToVaryRadiusByAbsoluteScalar()
        tubeFilter.SetNumberOfSides(16)
        tubeFilter.CappingOn()
        tubeFilter.Update()
        # smooth
        smoothFilter = vtk.vtkSmoothPolyDataFilter()
        smoothFilter.SetInputData(tubeFilter.GetOutput())
        smoothFilter.SetNumberOfIterations(2)
        smoothFilter.SetRelaxationFactor(0.5)
        smoothFilter.FeatureEdgeSmoothingOff()
        smoothFilter.BoundarySmoothingOn()
        smoothFilter.Update()
        # update
        tubeModelNode = slicer.util.getNode(self.featuresTubeModelNodeID)
        tubeModelNode.SetAndObservePolyData(smoothFilter.GetOutput())
        tubeModelNode.GetDisplayNode().SetActiveScalarName('Color')
        tubeModelNode.GetDisplayNode().SetAutoScalarRange(False)
        tubeModelNode.GetDisplayNode().SetScalarRange(0.0, 1.0)
        tubeModelNode.Modified()

    def updateMarkupsFromValues(self, samplePoints, markupsSize, markupsColor):
        pass  # TODO

    # @classmethod
    # def InitOrGetNthTrajectory(cls, trajectoryNumber):
    #     trajectory = cls.GetNthTrajectory(trajectoryNumber)
    #     if trajectory is not None:
    #         return trajectory
    #     else:
    #         return cls(trajectoryNumber)

    # @classmethod
    # def GetNthTrajectory(cls, trajectoryNumber):
    #     folderID = cls.GetFolderIDForNthTrajectory(trajectoryNumber)
    #     if folderID is not None:
    #         return cls(folderID, fromFolderID=True)

    # @classmethod
    # def GetTrajectoryFromChannelName(cls, channelName):
    #     folderID = cls.GetFolderIDForChannelName(channelName)
    #     if folderID is not None:
    #         return cls(folderID, fromFolderID=True)

    @classmethod
    def RemoveNthTrajectory(cls, trajectoryNumber):
        folderID = cls.GetFolderIDForNthTrajectory(trajectoryNumber)
        if folderID is not None:
            shNode = slicer.mrmlScene.GetSubjectHierarchyNode()
            shNode.RemoveItemChildren(folderID)
            shNode.RemoveItem(folderID)

    @staticmethod
    def GetFolderIDForNthTrajectory(N):
        shNode = slicer.mrmlScene.GetSubjectHierarchyNode()
        vtk_ids = vtk.vtkIdList()
        shNode.GetItemChildren(shNode.GetSceneItemID(), vtk_ids)
        IDs = [vtk_ids.GetId(i) for i in range(vtk_ids.GetNumberOfIds())]
        for ID in IDs:
            if 'LeadORTrajectory' in shNode.GetItemAttributeNames(ID):
                if int(shNode.GetItemAttribute(ID, 'LeadORTrajectory')) == N:
                    return ID

    @staticmethod
    def GetFolderIDForChannelName(channelName):
        shNode = slicer.mrmlScene.GetSubjectHierarchyNode()
        vtk_ids = vtk.vtkIdList()
        shNode.GetItemChildren(shNode.GetSceneItemID(), vtk_ids)
        IDs = [vtk_ids.GetId(i) for i in range(vtk_ids.GetNumberOfIds())]
        for ID in IDs:
            if 'LeadORTrajectory' in shNode.GetItemAttributeNames(ID):
                if shNode.GetItemAttribute(ID, 'ChannelName') == channelName:
                    return ID


class LeadORLogic(ScriptedLoadableModuleLogic):
    """This class should implement all the actual
    computation done by your module.  The interface
    should be such that other python code can import
    this class and make use of the functionality without
    requiring an instance of the Widget.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self):
        """
        Called when the logic class is instantiated. Can be used for initializing member variables.
        """
        ScriptedLoadableModuleLogic.__init__(self)

        self.VTASource = None

    def setDefaultParameters(self, parameterNode):
        """
        Initialize parameter node with default settings.
        """
        if not parameterNode.GetParameter("UnlinkedChannels"):
            parameterNode.SetParameter("UnlinkedChannels", "")
        if not parameterNode.GetParameter("FeatureNames"):
            parameterNode.SetParameter("FeatureNames", "")
        if not parameterNode.GetParameter("FeaturesJson"):
            parameterNode.SetParameter("FeaturesJson", json.dumps([]))
        if not parameterNode.GetParameter("TrajectoriesJson"):
            parameterNode.SetParameter("TrajectoriesJson", json.dumps(
                [{"active": 0, "channelName": '', "modelVisibility": 1, "lineVisibility": 1, "tipVisibility": 1}] * 9))

    def setUpTrajectory(self, trajectoryNumber, entry_target: EntryTarget, electrode_records,active=True, channelName='',
                        modelVisibility=1, lineVisibility=1, tipVisibility=1):
        if active:
            trajectory = Trajectory(entry_target, electrode_records)
            # trajectory.setDistanceToTargetTransformID(distanceToTargetTransformID)
            trajectory.setChannelName(channelName)
            trajectory.setModelVisibility(modelVisibility)
            trajectory.setLineVisibility(lineVisibility)
            trajectory.setTipVisibility(tipVisibility)
        else:
            Trajectory.RemoveNthTrajectory(trajectoryNumber)

    def setUpFeature(self, sourceNodeID=None, name='', projectTo='Tube', property='', visible=1):
        if sourceNodeID is None:
            return
        feature = Feature(projectTo)
        feature.addSourceNode(sourceNodeID, property, visible)
        feature.update()

    def getVTARadius(self, I, pw=60):
        # I: amplitude in Ampere
        # pw: pulse width in micro seconds
        # returns radius in meter
        from numpy import sqrt
        return ((pw / 90) ** 0.3) * sqrt(0.8 * I / 165)  # 0.72

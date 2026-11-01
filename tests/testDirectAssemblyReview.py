"""Independent coordinate and accessory checks for direct assembly integration."""

import numpy as np
from scipy.spatial.distance import pdist

from faceAppearance.correspondence import sampleSurface
from faceAppearance.directTemplateRegistration import registerDirectTemplate
from faceAppearance.headAssembly import AssemblyAssets
from testDirectAssembly import directBundle
from testHeadAssembly import bundle, reconstructedGeometry


def testAssemblyUsesItsOwnSurfaceSimilarityAndPreservesRigidAccessories(directBundle):
    root, arrays = directBundle
    assets = AssemblyAssets(root)
    geometry = reconstructedGeometry(arrays)
    output, components, diagnostics = assets._deform(geometry)
    mapping = assets.directArrays
    targets = sampleSurface(geometry['vertices'], geometry['faces'], {
        'triangleIds': mapping['sourceTriangleIds'],
        'barycentricWeights': mapping['sourceBarycentricWeights']})
    ids = arrays['nicpToFullIds']
    result = registerDirectTemplate(arrays['vertices'][ids], arrays['nicpFaces'],
        mapping['sampleNicpFaceIds'], mapping['sampleNicpWeights'], targets,
        mapping['sampleWeights'] > 0, np.flatnonzero(arrays['staticMask']),
        restWeight=0., smoothWeight=30., staticConstraint='hard')
    transform = result['transform']
    expected = ((result['rawVertices'] - transform['translation'])
                @ transform['rotation'].T) / transform['scale']
    expected[arrays['staticMask']] = arrays['vertices'][ids][arrays['staticMask']]
    np.testing.assert_array_equal(output['vertices'][ids], expected)
    for key in ('rotation', 'translation', 'scale'):
        np.testing.assert_array_equal(diagnostics['templateToUff'][key], transform[key])
    for component in components:
        part = component['vertexIndices']
        if component['transformAppliesToComponent']:
            matrix = component['transformRow4x4']
            predicted = arrays['vertices'][part] @ matrix[:3, :3] + matrix[3, :3]
            np.testing.assert_allclose(output['vertices'][part], predicted, atol=1e-14)
            np.testing.assert_allclose(pdist(output['vertices'][part]),
                                       pdist(arrays['vertices'][part]), atol=1e-14)
            np.testing.assert_allclose(np.linalg.det(matrix[:3, :3]), 1., atol=1e-14)
    assert diagnostics['geometryTransfer']['staticDriftMaxCm'] == 0.
    assert diagnostics['targetResidualPopulation'].startswith('Client triangle surface samples')


def testAssemblyCancelsInputGlobalSimilarityInClientFrame(directBundle):
    root, arrays = directBundle
    assets = AssemblyAssets(root)
    first = reconstructedGeometry(arrays)
    angle = .42
    rotation = np.array([[1., 0., 0.], [0., np.cos(angle), -np.sin(angle)],
                         [0., np.sin(angle), np.cos(angle)]])
    second = {**first, 'vertices': first['vertices'] @ rotation * 3.4 + [.9, .2, -.8]}
    original, _, _ = assets._deform(first)
    transformed, _, _ = assets._deform(second)
    np.testing.assert_allclose(transformed['vertices'], original['vertices'], atol=2e-12, rtol=2e-12)


def testDirectDoesNotCallPreloadedDisplacementSolver(directBundle):
    root, arrays = directBundle
    assets = AssemblyAssets(root)
    def forbidden(*args, **kwargs):
        raise AssertionError('Legacy UV displacement solve called')
    assets.solveDisplacement = forbidden
    assets._deform(reconstructedGeometry(arrays))

"""Choose one current-photo reference for client geometry and appearance.

The photo-only gate reuses the existing jaw photo reliability checks before
any client deformation. Later geometry/landmark stages keep their own gates;
they may skip a correction, but must not silently select another photograph.
"""
from __future__ import annotations

import re

import numpy as np

from .adaptiveChinRefinement import _prepare_views
from .clientChinRefinement import _json
from .geometryRefinement import _validateCameras


METHOD = 'sharedReliablePhotoReferenceV1'


def validateReferenceObservationIdentity(observations, inputs):
    """Reject stale or reordered cached evidence before using its landmarks."""
    records = observations.get('diagnostics', {}).get('views', [])
    if len(records) != len(inputs):
        raise ValueError('Reference photo observations require one identity record per input')
    identities = []
    for index, (record, metadata) in enumerate(zip(records, inputs)):
        digest = metadata.get('sha256')
        if (not isinstance(digest, str) or re.fullmatch(r'[a-f0-9]{64}', digest) is None
                or record.get('viewIndex') != index or record.get('photoSha256') != digest):
            raise ValueError('Reference photo observations do not match the current ordered input identities')
        identities.append(digest)
    return identities


def chooseReferenceReport(viewReports, photoIdentities, *, referenceViewIndex=None):
    """Select an eligible score winner, or preserve an explicit caller choice.

    A caller override takes precedence over automatic ranking. It does not
    bypass downstream anatomical, camera or geometry safety checks. Ties use
    the immutable photograph SHA, so permuting input order preserves identity.
    """
    count = len(photoIdentities)
    if not count or len(viewReports) != count:
        raise ValueError('Reference reports and photo identities must have identical nonzero counts')
    if referenceViewIndex is not None and (isinstance(referenceViewIndex, (bool, np.bool_))
            or not isinstance(referenceViewIndex, (int, np.integer))
            or not 0 <= referenceViewIndex < count):
        raise ValueError('referenceViewIndex must be None or a valid original input index')
    if [row.get('viewIndex') for row in viewReports] != list(range(count)):
        raise ValueError('Reference reports must retain the original input order')
    eligible = [row for row in viewReports if row.get('referenceEligible', False)
                and np.isfinite(row.get('referenceSelectionScore', np.nan))]
    ranked = sorted(eligible, key=lambda row: (-float(row['referenceSelectionScore']),
                                             photoIdentities[row['viewIndex']], row['viewIndex']))
    automatic = ranked[0]['viewIndex'] if ranked else None
    chosen = int(referenceViewIndex) if referenceViewIndex is not None else automatic
    return {'method': METHOD, 'status': ('userOverride' if referenceViewIndex is not None
            else 'selected' if chosen is not None else 'noReliableReference'),
            'referenceViewIndex': chosen,
            'referencePhotoSha256': photoIdentities[chosen] if chosen is not None else None,
            'automaticReferenceViewIndex': automatic,
            'duplicatePhotoIdentityCount': count - len(set(photoIdentities)),
            'selectedPhotoPassedAutomaticGate': bool(chosen is not None
                and viewReports[chosen].get('referenceEligible', False)),
            'views': _json(viewReports),
            'fallback': ('Keep direct registration; skip reference-driven eye/mouth/lip/jaw and texture overrides'
                         if chosen is None else None),
            'policy': 'One immutable photo reference; explicit override first, otherwise existing photo reliability/symmetry score; SHA tie-break',
            'qualification': 'Photo-only eligibility; downstream anatomical and geometric gates remain mandatory'}


def selectClientReference(cameras, inputs, photoObservations, *, referenceViewIndex=None):
    """Run the existing reliable frontal-photo gate without a template prior."""
    count = len(inputs)
    if len(cameras) != count or not count:
        raise ValueError('Reference selection requires matching nonempty cameras and input metadata')
    if referenceViewIndex is not None and (isinstance(referenceViewIndex, (bool, np.bool_))
            or not isinstance(referenceViewIndex, (int, np.integer))
            or not 0 <= referenceViewIndex < count):
        raise ValueError('referenceViewIndex must be None or a valid original input index')
    identities = validateReferenceObservationIdentity(photoObservations, inputs)
    checked, raw, invalid = [], [], {}
    for index, camera in enumerate(cameras):
        try:
            value = _validateCameras([camera])[0]
            if not camera.get('valid', True):
                raise ValueError('camera_not_accepted_by_reconstruction')
            checked.append(value)
            raw.append(camera)
        except (ValueError, TypeError, KeyError, np.linalg.LinAlgError) as error:
            # _prepare_views rejects this row before reading its placeholder
            # camera. Keep the row/index instead of silently dropping a view.
            checked.append({'intrinsics': np.eye(3), 'rotation': np.eye(3), 'translation': np.zeros(3)})
            raw.append({'valid': False})
            invalid[index] = str(error)
    _, reports = _prepare_views(np.empty((0, 3)), raw, checked, photoObservations, binding=None)
    for index, reason in invalid.items():
        reports[index].update(status='rejected', reason=reason, referenceEligible=False)
    if referenceViewIndex is not None and referenceViewIndex in invalid:
        raise ValueError('Explicit reference photograph has no accepted camera')
    return chooseReferenceReport(reports, identities, referenceViewIndex=referenceViewIndex)

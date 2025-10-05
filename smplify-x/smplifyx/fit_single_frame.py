# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems and the Max Planck Institute for Biological
# Cybernetics. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from kornia.geometry.conversions import rotation_matrix_to_angle_axis as rotmat_to_axis_angle



import time
try:
    import cPickle as pickle
except ImportError:
    import pickle

import sys
import os
import os.path as osp

import numpy as np
import torch

from tqdm import tqdm

from collections import defaultdict

import cv2
import PIL.Image as pil_img

from optimizers import optim_factory

import fitting
from human_body_prior.tools.model_loader import load_vposer

# from human_body_prior.tools.model_loader import load_model
# from human_body_prior.models.vposer_model import VPoser



def fit_single_frame(img,
                     keypoints,
                     body_model,
                     camera,
                     joint_weights,
                     body_pose_prior,
                     jaw_prior,
                     left_hand_prior,
                     right_hand_prior,
                     shape_prior,
                     expr_prior,
                     angle_prior,
                     result_fn='out.pkl',
                     mesh_fn='out.obj',
                     out_img_fn='overlay.png',
                     loss_type='smplify',
                     use_cuda=True,
                     init_joints_idxs=(9, 12, 2, 5),
                     use_face=True,
                     use_hands=True,
                     data_weights=None,
                     body_pose_prior_weights=None,
                     hand_pose_prior_weights=None,
                     jaw_pose_prior_weights=None,
                     shape_weights=None,
                     expr_weights=None,
                     hand_joints_weights=None,
                     face_joints_weights=None,
                     depth_loss_weight=1e2,
                     interpenetration=True,
                     coll_loss_weights=None,
                     df_cone_height=0.5,
                     penalize_outside=True,
                     max_collisions=8,
                     point2plane=False,
                     part_segm_fn='',
                     focal_length=5000.,
                     side_view_thsh=25.,
                     rho=100,
                     vposer_latent_dim=32,
                     vposer_ckpt='',
                     use_joints_conf=False,
                     interactive=True,
                     visualize=False,
                     save_meshes=True,
                     degrees=None,
                     batch_size=1,
                     dtype=torch.float32,
                     ign_part_pairs=None,
                     left_shoulder_idx=2,
                     right_shoulder_idx=5,
                     **kwargs):
    assert batch_size == 1, 'PyTorch L-BFGS only supports batch_size == 1'

    device = torch.device('cuda') if use_cuda else torch.device('cpu')

    if degrees is None:
        degrees = [0, 90, 180, 270]

    if data_weights is None:
        data_weights = [1, ] * 5

    if body_pose_prior_weights is None:
        body_pose_prior_weights = [4.04 * 1e2, 4.04 * 1e2, 57.4, 4.78]

    msg = (
        'Number of Body pose prior weights {}'.format(
            len(body_pose_prior_weights)) +
        ' does not match the number of data term weights {}'.format(
            len(data_weights)))
    assert (len(data_weights) ==
            len(body_pose_prior_weights)), msg

    if use_hands:
        if hand_pose_prior_weights is None:
            hand_pose_prior_weights = [1e2, 5 * 1e1, 1e1, .5 * 1e1]
        msg = ('Number of Body pose prior weights does not match the' +
               ' number of hand pose prior weights')
        assert (len(hand_pose_prior_weights) ==
                len(body_pose_prior_weights)), msg
        if hand_joints_weights is None:
            hand_joints_weights = [0.0, 0.0, 0.0, 1.0]
            msg = ('Number of Body pose prior weights does not match the' +
                   ' number of hand joint distance weights')
            assert (len(hand_joints_weights) ==
                    len(body_pose_prior_weights)), msg

    if shape_weights is None:
        shape_weights = [1e2, 5 * 1e1, 1e1, .5 * 1e1]
    msg = ('Number of Body pose prior weights = {} does not match the' +
           ' number of Shape prior weights = {}')
    assert (len(shape_weights) ==
            len(body_pose_prior_weights)), msg.format(
                len(shape_weights),
                len(body_pose_prior_weights))

    if use_face:
        if jaw_pose_prior_weights is None:
            jaw_pose_prior_weights = [[x] * 3 for x in shape_weights]
        else:
            jaw_pose_prior_weights = map(lambda x: map(float, x.split(',')),
                                         jaw_pose_prior_weights)
            jaw_pose_prior_weights = [list(w) for w in jaw_pose_prior_weights]
        msg = ('Number of Body pose prior weights does not match the' +
               ' number of jaw pose prior weights')
        assert (len(jaw_pose_prior_weights) ==
                len(body_pose_prior_weights)), msg

        if expr_weights is None:
            expr_weights = [1e2, 5 * 1e1, 1e1, .5 * 1e1]
        msg = ('Number of Body pose prior weights = {} does not match the' +
               ' number of Expression prior weights = {}')
        assert (len(expr_weights) ==
                len(body_pose_prior_weights)), msg.format(
                    len(body_pose_prior_weights),
                    len(expr_weights))

        if face_joints_weights is None:
            face_joints_weights = [0.0, 0.0, 0.0, 1.0]
        msg = ('Number of Body pose prior weights does not match the' +
               ' number of face joint distance weights')
        assert (len(face_joints_weights) ==
                len(body_pose_prior_weights)), msg

    if coll_loss_weights is None:
        coll_loss_weights = [0.0] * len(body_pose_prior_weights)
    msg = ('Number of Body pose prior weights does not match the' +
           ' number of collision loss weights')
    assert (len(coll_loss_weights) ==
            len(body_pose_prior_weights)), msg

    use_vposer = kwargs.get('use_vposer', True)
    vposer, pose_embedding = [None, ] * 2
    if use_vposer:
        pose_embedding = torch.zeros([batch_size, 32],
                                     dtype=dtype, device=device,
                                     requires_grad=True)

        vposer_ckpt = osp.expandvars(vposer_ckpt)
        vposer, _ = load_vposer(vposer_ckpt, vp_model='snapshot')
        vposer = vposer.to(device=device)
        vposer.eval()
    
    # use_vposer = kwargs.get('use_vposer', True)
    # vposer, pose_embedding = [None, ] * 2
    # if use_vposer:
    #     pose_embedding = torch.zeros([batch_size, vposer_latent_dim],
    #                                 dtype=dtype, device=device, requires_grad=True)

    #     vposer_ckpt = osp.expandvars(vposer_ckpt)
    #     # If user passed a file path by mistake, turn it into the experiment dir
    #     if osp.isfile(vposer_ckpt):
    #         vposer_ckpt = osp.dirname(osp.dirname(vposer_ckpt))  # go up to ...\vposer_v1_0
            
    #     snap_dir = osp.join(vposer_ckpt, 'snapshots')
    #     print('[VPoser] expr_dir =', osp.abspath(vposer_ckpt))
    #     print('[VPoser] snapshots =', osp.abspath(snap_dir))
    #     print('[VPoser] files:', os.listdir(snap_dir) if osp.isdir(snap_dir) else 'NO DIR')

    #     # human_body_prior.load_model for your version: NO map_location/remove_words kw
    #     vposer, _ = load_model(vposer_ckpt, model_code=VPoser)
    #     vposer = vposer.to(device=device).eval()


    # if use_vposer:
    #     body_mean_pose = torch.zeros([batch_size, vposer_latent_dim],
    #                                  dtype=dtype)
    # else:
    #     body_mean_pose = body_pose_prior.get_mean().detach().cpu()
        
    if use_vposer:
        body_mean_pose = torch.zeros([batch_size, vposer_latent_dim], dtype=dtype)
    else:
        # L2Prior has no get_mean(); use zeros with the correct dimension
        if hasattr(body_model, 'body_pose') and hasattr(body_model.body_pose, 'shape'):
            n_pose = int(body_model.body_pose.shape[-1])  # usually 63 for SMPL/SMPL-X body
        else:
            n_pose = 63  # safe fallback (21 body joints * 3 axis-angle)
        body_mean_pose = torch.zeros([batch_size, n_pose], dtype=dtype)


    keypoint_data = torch.tensor(keypoints, dtype=dtype)
    gt_joints = keypoint_data[:, :, :2]
    joints_conf = None
    if use_joints_conf:
        joints_conf = keypoint_data[:, :, 2].reshape(1, -1)

    # Transfer the data to the correct device
    gt_joints = gt_joints.to(device=device, dtype=dtype)
    if use_joints_conf:
        joints_conf = joints_conf.to(device=device, dtype=dtype)

    # Create the search tree
    search_tree = None
    pen_distance = None
    filter_faces = None
    if interpenetration:
        from mesh_intersection.bvh_search_tree import BVH
        import mesh_intersection.loss as collisions_loss
        from mesh_intersection.filter_faces import FilterFaces

        assert use_cuda, 'Interpenetration term can only be used with CUDA'
        assert torch.cuda.is_available(), \
            'No CUDA Device! Interpenetration term can only be used' + \
            ' with CUDA'

        search_tree = BVH(max_collisions=max_collisions)

        pen_distance = \
            collisions_loss.DistanceFieldPenetrationLoss(
                sigma=df_cone_height, point2plane=point2plane,
                vectorized=True, penalize_outside=penalize_outside)

        if part_segm_fn:
            # Read the part segmentation
            part_segm_fn = os.path.expandvars(part_segm_fn)
            with open(part_segm_fn, 'rb') as faces_parents_file:
                face_segm_data = pickle.load(faces_parents_file,
                                             encoding='latin1')
            faces_segm = face_segm_data['segm']
            faces_parents = face_segm_data['parents']
            # Create the module used to filter invalid collision pairs
            filter_faces = FilterFaces(
                faces_segm=faces_segm, faces_parents=faces_parents,
                ign_part_pairs=ign_part_pairs).to(device=device)

    # Weights used for the pose prior and the shape prior
    opt_weights_dict = {'data_weight': data_weights,
                        'body_pose_weight': body_pose_prior_weights,
                        'shape_weight': shape_weights}
    if use_face:
        opt_weights_dict['face_weight'] = face_joints_weights
        opt_weights_dict['expr_prior_weight'] = expr_weights
        opt_weights_dict['jaw_prior_weight'] = jaw_pose_prior_weights
    if use_hands:
        opt_weights_dict['hand_weight'] = hand_joints_weights
        opt_weights_dict['hand_prior_weight'] = hand_pose_prior_weights
    if interpenetration:
        opt_weights_dict['coll_loss_weight'] = coll_loss_weights

    keys = opt_weights_dict.keys()
    opt_weights = [dict(zip(keys, vals)) for vals in
                   zip(*(opt_weights_dict[k] for k in keys
                         if opt_weights_dict[k] is not None))]
    for weight_list in opt_weights:
        for key in weight_list:
            weight_list[key] = torch.tensor(weight_list[key],
                                            device=device,
                                            dtype=dtype)

    # The indices of the joints used for the initialization of the camera
    init_joints_idxs = torch.tensor(init_joints_idxs, device=device)

    edge_indices = kwargs.get('body_tri_idxs')
    init_t = fitting.guess_init(body_model, gt_joints, edge_indices,
                                use_vposer=use_vposer, vposer=vposer,
                                pose_embedding=pose_embedding,
                                model_type=kwargs.get('model_type', 'smpl'),
                                focal_length=focal_length, dtype=dtype)

    camera_loss = fitting.create_loss('camera_init',
                                      trans_estimation=init_t,
                                      init_joints_idxs=init_joints_idxs,
                                      depth_loss_weight=depth_loss_weight,
                                      dtype=dtype).to(device=device)
    camera_loss.trans_estimation[:] = init_t

    loss = fitting.create_loss(loss_type=loss_type,
                               joint_weights=joint_weights,
                               rho=rho,
                               use_joints_conf=use_joints_conf,
                               use_face=use_face, use_hands=use_hands,
                               vposer=vposer,
                               pose_embedding=pose_embedding,
                               body_pose_prior=body_pose_prior,
                               shape_prior=shape_prior,
                               angle_prior=angle_prior,
                               expr_prior=expr_prior,
                               left_hand_prior=left_hand_prior,
                               right_hand_prior=right_hand_prior,
                               jaw_prior=jaw_prior,
                               interpenetration=interpenetration,
                               pen_distance=pen_distance,
                               search_tree=search_tree,
                               tri_filtering_module=filter_faces,
                               dtype=dtype,
                               **kwargs)
    loss = loss.to(device=device)

    with fitting.FittingMonitor(
            batch_size=batch_size, visualize=visualize, **kwargs) as monitor:

        img = torch.tensor(img, dtype=dtype)

        H, W, _ = img.shape

        data_weight = 1000 / H
        # The closure passed to the optimizer
        camera_loss.reset_loss_weights({'data_weight': data_weight})

        # Reset the parameters to estimate the initial translation of the
        # body model
   
        if use_vposer:
            body_model.reset_params()          # <-- no body_pose here when using VPoser
        else:
            body_model.reset_params(body_pose=body_mean_pose)
        
        # body_model.reset_params(body_pose=body_mean_pose)

        # If the distance between the 2D shoulders is smaller than a
        # predefined threshold then try 2 fits, the initial one and a 180
        # degree rotation
        shoulder_dist = torch.dist(gt_joints[:, left_shoulder_idx],
                                   gt_joints[:, right_shoulder_idx])
        try_both_orient = shoulder_dist.item() < side_view_thsh

        # Update the value of the translation of the camera as well as
        # the image center.
        with torch.no_grad():
            camera.translation[:] = init_t.view_as(camera.translation)
            camera.center[:] = torch.tensor([W, H], dtype=dtype) * 0.5

        # Re-enable gradient calculation for the camera translation
        camera.translation.requires_grad = True

        camera_opt_params = [camera.translation, body_model.global_orient]

        camera_optimizer, camera_create_graph = optim_factory.create_optimizer(
            camera_opt_params,
            **kwargs)

        # The closure passed to the optimizer
        fit_camera = monitor.create_fitting_closure(
            camera_optimizer, body_model, camera, gt_joints,
            camera_loss, create_graph=camera_create_graph,
            use_vposer=use_vposer, vposer=vposer,
            pose_embedding=pose_embedding,
            return_full_pose=False, return_verts=False)

        # Step 1: Optimize over the torso joints the camera translation
        # Initialize the computational graph by feeding the initial translation
        # of the camera and the initial pose of the body model.
        camera_init_start = time.time()
        cam_init_loss_val = monitor.run_fitting(camera_optimizer,
                                                fit_camera,
                                                camera_opt_params, body_model,
                                                use_vposer=use_vposer,
                                                pose_embedding=pose_embedding,
                                                vposer=vposer)

        if interactive:
            if use_cuda and torch.cuda.is_available():
                torch.cuda.synchronize()
            tqdm.write('Camera initialization done after {:.4f}'.format(
                time.time() - camera_init_start))
            tqdm.write('Camera initialization final loss {:.4f}'.format(
                cam_init_loss_val))

        # If the 2D detections/positions of the shoulder joints are too
        # close the rotate the body by 180 degrees and also fit to that
        # orientation
        if try_both_orient:
            body_orient = body_model.global_orient.detach().cpu().numpy()
            flipped_orient = cv2.Rodrigues(body_orient)[0].dot(
                cv2.Rodrigues(np.array([0., np.pi, 0]))[0])
            flipped_orient = cv2.Rodrigues(flipped_orient)[0].ravel()

            flipped_orient = torch.tensor(flipped_orient,
                                          dtype=dtype,
                                          device=device).unsqueeze(dim=0)
            orientations = [body_orient, flipped_orient]
        else:
            orientations = [body_model.global_orient.detach().cpu().numpy()]

        # store here the final error for both orientations,
        # and pick the orientation resulting in the lowest error
        results = []

        # Step 2: Optimize the full model
        final_loss_val = 0
        for or_idx, orient in enumerate(tqdm(orientations, desc='Orientation')):
            opt_start = time.time()

            new_params = defaultdict(global_orient=orient,
                                     body_pose=body_mean_pose)
            body_model.reset_params(**new_params)
            if use_vposer:
                with torch.no_grad():
                    pose_embedding.fill_(0)

            for opt_idx, curr_weights in enumerate(tqdm(opt_weights, desc='Stage')):

                body_params = list(body_model.parameters())

                final_params = list(
                    filter(lambda x: x.requires_grad, body_params))

                if use_vposer:
                    final_params.append(pose_embedding)

                body_optimizer, body_create_graph = optim_factory.create_optimizer(
                    final_params,
                    **kwargs)
                body_optimizer.zero_grad()

                curr_weights['data_weight'] = data_weight
                curr_weights['bending_prior_weight'] = (
                    3.17 * curr_weights['body_pose_weight'])
                if use_hands:
                    joint_weights[:, 25:67] = curr_weights['hand_weight']
                if use_face:
                    joint_weights[:, 67:] = curr_weights['face_weight']
                loss.reset_loss_weights(curr_weights)

                closure = monitor.create_fitting_closure(
                    body_optimizer, body_model,
                    camera=camera, gt_joints=gt_joints,
                    joints_conf=joints_conf,
                    joint_weights=joint_weights,
                    loss=loss, create_graph=body_create_graph,
                    use_vposer=use_vposer, vposer=vposer,
                    pose_embedding=pose_embedding,
                    return_verts=True, return_full_pose=True)

                if interactive:
                    if use_cuda and torch.cuda.is_available():
                        torch.cuda.synchronize()
                    stage_start = time.time()
                final_loss_val = monitor.run_fitting(
                    body_optimizer,
                    closure, final_params,
                    body_model,
                    pose_embedding=pose_embedding, vposer=vposer,
                    use_vposer=use_vposer)

                # global_orient = final_params['global_orient']
                # body_pose = final_params['body_pose']
                # betas = final_params['betas']
                # transl = final_params['transl']
                # expression = final_params['expression']
                # jaw_pose = final_params['jaw_pose']
                # leye_pose = final_params['leye_pose']
                # reye_pose = final_params['reye_pose']
                # left_hand_pose = final_params['left_hand_pose']
                # right_hand_pose = final_params['right_hand_pose']
                
                if interactive:
                    if use_cuda and torch.cuda.is_available():
                        torch.cuda.synchronize()
                    elapsed = time.time() - stage_start
                    if interactive:
                        tqdm.write('Stage {:03d} done after {:.4f} seconds'.format(
                            opt_idx, elapsed))

            if interactive:
                if use_cuda and torch.cuda.is_available():
                    torch.cuda.synchronize()
                elapsed = time.time() - opt_start
                
                tqdm.write(
                    'Body fitting Orientation {} done after {:.4f} seconds'.format(
                        or_idx, elapsed))
                if isinstance(final_loss_val, (float, int)):
                    tqdm.write('Body final loss val = {:.5f}'.format(
                    final_loss_val))
                else:
                    tqdm.write(f'Body final loss val = {final_loss_val}')

            # Get the result of the fitting process
            # Store in it the errors list in order to compare multiple
            # orientations, if they exist
            result = {'camera_' + str(key): val.detach().cpu().numpy()
                      for key, val in camera.named_parameters()}
            result.update({key: val.detach().cpu().numpy()
                           for key, val in body_model.named_parameters()})
            if use_vposer:
                result['body_pose'] = pose_embedding.detach().cpu().numpy()

            results.append({'loss': final_loss_val,
                            'result': result})

        with open(result_fn, 'wb') as result_file:
            if len(results) > 1:
                min_idx = (0 if results[0]['loss'] < results[1]['loss']
                           else 1)
            else:
                min_idx = 0
            pickle.dump(results[min_idx]['result'], result_file, protocol=2)

    # if save_meshes or visualize:
    #     if use_vposer:
    #         decoded = vposer.decode(pose_embedding)
    #         # VPoser v2 returns a dict with 'pose_body'; v1 returns a tensor
    #         if isinstance(decoded, dict) and 'pose_body' in decoded:
    #             body_pose = decoded['pose_body'].reshape(1, -1)
    #         else:
    #             body_pose = decoded.view(1, -1)
    #     else:
    #         body_pose = None

        if save_meshes or visualize:
            # We will produce body_pose as (B, 21, 3) axis–angle
            body_pose = None

            if use_vposer:
                decoded = vposer.decode(pose_embedding)  # 'aa' not always supported
                bp = decoded['pose_body'] if (isinstance(decoded, dict) and 'pose_body' in decoded) else decoded
                bp = bp.to(device=device, dtype=dtype)

                # Squeeze stray singleton dims, e.g. (B,1,21,9) -> (B,21,9)
                while bp.dim() > 2 and bp.size(1) == 1:
                    bp = bp.squeeze(1)

                if bp.dim() == 2 and bp.shape[-1] == 63:
                    # (B,63) axis–angle packed -> (B,21,3)
                    body_pose = bp.view(bp.size(0), 21, 3)

                elif bp.dim() == 3 and bp.shape[1:] == (21, 3):
                    # Already (B,21,3) axis–angle
                    body_pose = bp

                elif (bp.dim() == 2 and bp.shape[-1] == 189) or \
                    (bp.dim() == 3 and bp.shape[1:] == (21, 9)) or \
                    (bp.dim() == 4 and bp.shape[-2:] == (3, 3)):
                    # Rotation matrices -> axis–angle using smplx util
                    # from smplx.lbs import batch_rot2aa
                    rotmats = bp.view(-1, 3, 3)       # (B*21,3,3)
                    aa = rotmat_to_axis_angle(rotmats)        # (B*21,3)
                    body_pose = aa.view(-1, 21, 3)    # (B,21,3)

                else:
                    raise ValueError(
                        f"Unexpected VPoser decode shape {tuple(bp.shape)}; "
                        "expected (B,63)|(B,21,3)|(B,189)|(B,21,9)|(B,21,3,3)"
                    )
            else:
                # No VPoser: fall back to mean pose if it’s 63-long, else keep None
                if body_mean_pose.numel() == 63:
                    body_pose = body_mean_pose.view(1, 21, 3)

        # Now call the model
        model_output = body_model(return_verts=True, body_pose=body_pose)
    # with torch.no_grad():
    #     # 1. Get the model output with the final optimized parameters
    #     model_output = body_model(global_orient=global_orient,
    #                             body_pose=body_pose,
    #                             betas=betas,
    #                             transl=transl,
    #                             expression=expression,
    #                             jaw_pose=jaw_pose,
    #                             leye_pose=leye_pose,
    #                             reye_pose=reye_pose,
    #                             left_hand_pose=left_hand_pose,
    #                             right_hand_pose=right_hand_pose)
    #     # 2. Calculate the final loss tensor by calling the loss object
    #     final_loss_values = loss(model_output, gt_joints, body_model_faces,
    #                             joint_conf, joint_weights, **kwargs)

    #     min_idx = torch.argmin(final_loss_values)

    #     # Extract the final, best-fit parameters
    #     body_pose = vposer_model.decode(
    #         pose_embedding[min_idx:min_idx+1], output_type='aa')

    #     global_orient = global_orient[min_idx:min_idx+1]
    #     betas = betas[min_idx:min_idx+1]
    #     transl = transl[min_idx:min_idx+1]
    #     expression = expression[min_idx:min_idx+1]
    #     jaw_pose = jaw_pose[min_idx:min_idx+1]
    #     leye_pose = leye_pose[min_idx:min_idx+1]
    #     reye_pose = reye_pose[min_idx:min_idx+1]
    #     left_hand_pose = left_hand_pose[min_idx:min_idx+1]
    #     right_hand_pose = right_hand_pose[min_idx:min_idx+1]

    #     # Call the model with all the selected parameters to get the final mesh
    #     model_output = body_model(return_verts=True,
    #                             global_orient=global_orient,
    #                             body_pose=body_pose,
    #                             betas=betas,
    #                             transl=transl,
    #                             expression=expression,
    #                             jaw_pose=jaw_pose,
    #                             leye_pose=leye_pose,
    #                             reye_pose=reye_pose,
    #                             left_hand_pose=left_hand_pose,
    #                             right_hand_pose=right_hand_pose)


        vertices = model_output.vertices.detach().cpu().numpy().squeeze()

        import trimesh

        out_mesh = trimesh.Trimesh(vertices, body_model.faces, process=False)
        rot = trimesh.transformations.rotation_matrix(
            np.radians(180), [1, 0, 0])
        out_mesh.apply_transform(rot)
        out_mesh.export(mesh_fn)

    # if visualize:
    #     import pyrender

    #     material = pyrender.MetallicRoughnessMaterial(
    #         metallicFactor=0.0,
    #         alphaMode='OPAQUE',
    #         baseColorFactor=(1.0, 1.0, 0.9, 1.0))
    #     mesh = pyrender.Mesh.from_trimesh(
    #         out_mesh,
    #         material=material)

    #     scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0],
    #                            ambient_light=(0.3, 0.3, 0.3))
    #     scene.add(mesh, 'mesh')

    #     camera_center = camera.center.detach().cpu().numpy().squeeze()
    #     camera_transl = camera.translation.detach().cpu().numpy().squeeze()
    #     # Equivalent to 180 degrees around the y-axis. Transforms the fit to
    #     # OpenGL compatible coordinate system.
    #     camera_transl[0] *= -1.0

    #     camera_pose = np.eye(4)
    #     camera_pose[:3, 3] = camera_transl

    #     camera = pyrender.camera.IntrinsicsCamera(
    #         fx=focal_length, fy=focal_length,
    #         cx=camera_center[0], cy=camera_center[1])
    #     scene.add(camera, pose=camera_pose)

    #     # Get the lights from the viewer
    #     light_nodes = monitor.mv.viewer._create_raymond_lights()
    #     for node in light_nodes:
    #         scene.add_node(node)

    #     r = pyrender.OffscreenRenderer(viewport_width=W,
    #                                    viewport_height=H,
    #                                    point_size=1.0)
    #     color, _ = r.render(scene, flags=pyrender.RenderFlags.RGBA)
    #     color = color.astype(np.float32) / 255.0

    #     valid_mask = (color[:, :, -1] > 0)[:, :, np.newaxis]
    #     input_img = img.detach().cpu().numpy()
    #     output_img = (color[:, :, :-1] * valid_mask +
    #                   (1 - valid_mask) * input_img)

    #     img = pil_img.fromarray((output_img * 255).astype(np.uint8))
    #     img.save(out_img_fn)
    
    if visualize:
        import pyrender, os

        try:
            material = pyrender.MetallicRoughnessMaterial(
                metallicFactor=0.0,
                alphaMode='OPAQUE',
                baseColorFactor=(1.0, 1.0, 0.9, 1.0))
            mesh = pyrender.Mesh.from_trimesh(out_mesh, material=material)

            scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0],
                                   ambient_light=(0.3, 0.3, 0.3))
            scene.add(mesh, 'mesh')

            camera_center = camera.center.detach().cpu().numpy().squeeze()
            camera_transl = camera.translation.detach().cpu().numpy().squeeze()
            camera_transl[0] *= -1.0  # OpenGL convention

            camera_pose = np.eye(4)
            camera_pose[:3, 3] = camera_transl

            cam = pyrender.camera.IntrinsicsCamera(
                fx=focal_length, fy=focal_length,
                cx=float(camera_center[0]), cy=float(camera_center[1]))
            scene.add(cam, pose=camera_pose)

            # Lights
            try:
                light_nodes = monitor.mv.viewer._create_raymond_lights()
                for node in light_nodes:
                    scene.add_node(node)
            except Exception:
                # Fallback simple lights if viewer helper not present
                scene.add(pyrender.DirectionalLight(), pose=np.eye(4))

            # Render
            H_int, W_int = int(H), int(W)
            os.makedirs(os.path.dirname(out_img_fn), exist_ok=True)

            r = pyrender.OffscreenRenderer(viewport_width=W_int,
                                           viewport_height=H_int,
                                           point_size=1.0)
            try:
                color, _ = r.render(scene, flags=pyrender.RenderFlags.RGBA)
            finally:
                r.delete()  # important on Windows to free the context

            color = color.astype(np.float32) / 255.0
            valid_mask = (color[:, :, -1] > 0)[:, :, np.newaxis]
            input_img = img.detach().cpu().numpy()
            output_img = (color[:, :, :-1] * valid_mask +
                          (1 - valid_mask) * input_img)

            vis_pil = pil_img.fromarray((output_img * 255).astype(np.uint8))
            vis_pil.save(out_img_fn)
            print(f"[vis] wrote {out_img_fn}")
        except Exception as e:
            print(f"[vis] failed to write overlay for this frame: {e}")

import re


def parse_references(text: str) -> dict:
    """
    Parse references from a given text and extract titles.

    Args:
    text (str): The input text containing references.

    Returns:
    dict: A dictionary where keys are reference numbers and values are titles.
    """
    parts = text.split("[")
    refs = {}

    for part in parts[1:]:  # Skip the first empty part
        try:
            number, content = part.split("]", 1)
            number = number.strip()

            # Split content by periods
            sentences = content.split(".")

            # Use the 2rd-to-last sentence as the title
            title = sentences[-3].strip()

            refs[number] = title
        except Exception as e:
            print(f"Error processing reference {number}: {str(e)}")
            refs[number] = content.strip()

    return refs


def replace_citations(text: str, refs: dict) -> str:
    """
    Replace citation numbers with their corresponding titles in the given text.

    Args:
    text (str): The input text containing citations.
    refs (dict): A dictionary of reference numbers and their titles.

    Returns:
    str: The text with citations replaced by titles.
    """
    # Handle multiple citations first
    for match in re.finditer(r"\[[\d,\s]+\]", text):
        citation = match.group()
        numbers = [num.strip() for num in citation[1:-1].split(",")]
        replaced = "[" + ", ".join(refs[num] for num in numbers if num in refs) + "]"
        text = text.replace(citation, replaced)

    # Then handle single citations
    for num, ref in refs.items():
        text = text.replace(f"[{num}]", f"[{ref}]")

    return text


if __name__ == "__main__":
    reference_text = """
    References [1] J. Czarnowski, T. Laidlow, R. Clark, and A. J. Davison. Deepfactors: Real-time probabilistic dense monocular SLAM. IEEE Robotics and Automation Letters (RAL), 5(2): 721–728, 2020. [2] Angela Dai, Matthias Nießner, Michael Zollh ̈ ofer, Shahram Izadi, and Christian Theobalt. BundleFusion: Real-time Globally Consistent 3D Reconstruction using On-the-fly Surface Re-integration. ACM Transactions on Graphics (TOG), 36(3):24:1–24:18, 2017. [3] Eric Dexheimer and Andrew J. Davison. Learning a Depth Covariance Function. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2023. [4] J. Engel, V. Koltun, and D. Cremers. Direct sparse odometry. IEEE Transactions on Pattern Analysis and Machine Intelligence (PAMI), 2017. [5] C. Forster, M. Pizzoli, and D. Scaramuzza. SVO: Fast SemiDirect Monocular Visual Odometry. In Proceedings of the IEEE International Conference on Robotics and Automation (ICRA), 2014. [6] Sara Fridovich-Keil, Alex Yu, Matthew Tancik, Qinhong Chen, Benjamin Recht, and Angjoo Kanazawa. Plenoxels: Radiance fields without neural networks. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2022. [7] Jiahui Huang, Shi-Sheng Huang, Haoxuan Song, and ShiMin Hu. Di-fusion: Online implicit 3d reconstruction with deep priors. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2021. [8] M. M. Johari, C. Carta, and F. Fleuret. ESLAM: Efficient dense slam system based on hybrid representation of signed distance fields. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2023. [9] M. Keller, D. Lefloch, M. Lambers, S. Izadi, T. Weyrich, and A. Kolb. Real-time 3D Reconstruction in Dynamic Scenes using Point-based Fusion. In Proc. of Joint 3DIM/3DPVT Conference (3DV), 2013. [10] Bernhard Kerbl, Georgios Kopanas, Thomas Leimk ̈ uhler, and George Drettakis. 3D gaussian splatting for real-time radiance field rendering. ACM Transactions on Graphics (TOG), 2023. [11] Leonid Keselman and Martial Hebert. Approximate differentiable rendering with algebraic surfaces. In Proceedings of the European Conference on Computer Vision (ECCV), 2022. [12] Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In Proceedings of the International Conference on Learning Representations (ICLR), 2015. [13] Heng Li, Xiaodong Gu, Weihao Yuan, Luwei Yang, Zilong Dong, and Ping Tan. Dense rgb slam with neural implicit maps. In Proceedings of the International Conference on Learning Representations (ICLR), 2023. [14] Lingjie Liu, Jiatao Gu, Kyaw Zaw Lin, Tat-Seng Chua, and Christian Theobalt. Neural sparse voxel fields. NeurIPS, 2020. [15] Jonathon Luiten, Georgios Kopanas, Bastian Leibe, and Deva Ramanan. Dynamic 3d gaussians: Tracking by persistent dynamic view synthesis. 3DV, 2024. [16] J. McCormac, A. Handa, A. J. Davison, and S. Leutenegger. SemanticFusion: Dense 3D semantic mapping with convolutional neural networks. In Proceedings of the IEEE International Conference on Robotics and Automation (ICRA), 2017. [17] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. In Proceedings of the European Conference on Computer Vision (ECCV), 2020. [18] N. J. Mitra, N. Gelfand, H. Pottmann, and L. J. Guibas. Registration of Point Cloud Data from a Geometric Optimization Perspective. In Proceedings of the Symposium on Geometry Processing, 2004. [19] Thomas M ̈ uller, Alex Evans, Christoph Schied, and Alexander Keller. Instant neural graphics primitives with a multiresolution hash encoding. ACM Transactions on Graphics (TOG), 2022. [20] R. Mur-Artal and J. D. Tard ́ os. ORB-SLAM2: An OpenSource SLAM System for Monocular, Stereo, and RGB-D Cameras. IEEE Transactions on Robotics (T-RO), 33(5): 1255–1262, 2017. [21] R. Mur-Artal, J. M. M Montiel, and J. D. Tard ́ os. ORBSLAM: a Versatile and Accurate Monocular SLAM System. IEEE Transactions on Robotics (T-RO), 31(5):1147–1163, 2015. [22] R. A. Newcombe. Dense Visual SLAM. PhD thesis, Imperial College London, 2012. [23] R. A. Newcombe, S. Izadi, O. Hilliges, D. Molyneaux, D. Kim, A. J. Davison, P. Kohli, J. Shotton, S. Hodges, and A. Fitzgibbon. KinectFusion: Real-Time Dense Surface Mapping and Tracking. In Proceedings of the International Symposium on Mixed and Augmented Reality (ISMAR), 2011. [24] Michael Niemeyer, Lars Mescheder, Michael Oechsle, and Andreas Geiger. Differentiable volumetric rendering: Learning implicit 3d representations without 3d supervision. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2020. [25] M. Nießner, M. Zollh ̈ofer, S. Izadi, and M. Stamminger. Real-time 3D Reconstruction at Scale using Voxel Hashing. In Proceedings of SIGGRAPH, 2013. [26] Victor Adrian Prisacariu, Olaf K ̈ ahler, Ming-Ming Cheng, Carl Yuheng Ren, Julien P. C. Valentin, Philip H. S. Torr, Ian D. Reid, and David W. Murray. A framework for the volumetric integration of depth images. CoRR, abs/1410.0925, 2014. [27] Erik Sandstr ̈ om, Yue Li, Luc Van Ghoul, and Martin R. Oswald. Point-slam: Dense neural point cloud-based slam. In Proceedings of the International Conference on Computer Vision (ICCV), 2023. [28] Thomas Sch ̈ ops, Torsten Sattler, and Marc Pollefeys. Surfelmeshing: Online surfel-based mesh reconstruction. IEEE Transactions on Pattern Analysis and Machine Intelligence (PAMI), 2020. [29] Thomas Sch ̈ops, Torsten Sattler, and Marc Pollefeys. Bad slam: Bundle adjusted direct rgb-d slam. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2019. [30] J. Sol a, J. Deray, and D. Atchuthan. A micro Lie theory for state estimation in robotics. arXiv:1812.01537, 2018. [31] Julian Straub, Thomas Whelan, Lingni Ma, Yufan Chen, Erik Wijmans, Simon Green, Jakob J. Engel, Raul Mur-Artal, Carl Ren, Shobhit Verma, Anton Clarkson, Mingfei Yan, Brian Budge, Yajie Yan, XiaqingPan, June Yon, Yuyang Zou, Kimberly Leon, Nigel Carter, Jesus Briales, Tyler Gillingham, Elias Mueggler, Luis Pesqueira, Manolis Savva, Dhruv Batra, Hauke M. Strasdat, Renzo De Nardi, Michael Goesele, Steven Lovegrove, and Richard Newcombe. The Replica dataset: A digital replica of indoor spaces. arXiv preprint arXiv:1906.05797, 2019. [32] J. Sturm, N. Engelhard, F. Endres, W. Burgard, and D. Cremers. A Benchmark for the Evaluation of RGB-D SLAM Systems. In Proceedings of the IEEE/RSJ Conference on Intelligent Robots and Systems (IROS), 2012. [33] E. Sucar, S. Liu, J. Ortiz, and A. J. Davison. iMAP: Implicit mapping and positioning in real-time. In Proceedings of the International Conference on Computer Vision (ICCV), 2021. [34] Cheng Sun, Min Sun, and Hwann-Tzong Chen. Direct voxel grid optimization: Super-fast convergence for radiance fields reconstruction. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2022. [35] Jiaxiang Tang, Jiawei Ren, Hang Zhou, Ziwei Liu, and Gang Zeng. Dreamgaussian: Generative gaussian splatting for efficient 3d content creation. Proceedings of the International Conference on Learning Representations (ICLR), 2024. [36] Zachary Teed and Jia Deng. DROID-SLAM: Deep Visual SLAM for Monocular, Stereo, and RGB-D Cameras. In Neural Information Processing Systems (NIPS), 2021. [37] Emanuele Vespa, Nikolay Nikolov, Marius Grimm, Luigi Nardi, Paul HJ Kelly, and Stefan Leutenegger. Efficient octree-based volumetric SLAM supporting signed-distance and occupancy mapping. IEEE Robotics and Automation Letters (RAL), 2018. [38] Angtian Wang, Peng Wang, Jian Sun, Adam Kortylewski, and Alan Yuille. Voge: a differentiable volume renderer using gaussian ellipsoids for analysis-by-synthesis. 2022. [39] Hengyi Wang, Jingwen Wang, and Lourdes Agapito. Coslam: Joint coordinate and sparse parametric encodings for neural real-time slam. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2023. [40] T. Whelan, M. Kaess, H. Johannsson, M. F. Fallon, J. J. Leonard, and J. B. McDonald. Real-time large scale dense RGB-D SLAM with volumetric fusion. International Journal of Robotics Research (IJRR), 34(4-5):598–626, 2015. [41] T. Whelan, S. Leutenegger, R. F. Salas-Moreno, B. Glocker, and A. J. Davison. ElasticFusion: Dense SLAM without a pose graph. In Proceedings of Robotics: Science and Systems (RSS), 2015. [42] Guanjun Wu, Taoran Yi, Jiemin Fang, Lingxi Xie, Xiaopeng Zhang, Wei Wei, Wenyu Liu, Qi Tian, and Xinggang Wang. 4d gaussian splatting for real-time dynamic scene rendering. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2024. [43] Xingrui Yang, Hai Li, Hongjia Zhai, Yuhang Ming, Yuqian Liu, and Guofeng Zhang. Vox-fusion: Dense tracking and mapping with voxel-based neural implicit representation. In Proceedings of the International Symposium on Mixed and Augmented Reality (ISMAR), 2022. [44] Zeyu Yang, Hongye Yang, Zijie Pan, Xiatian Zhu, and Li Zhang. Real-time photorealistic dynamic scene representation and rendering with 4d gaussian splatting. Proceedings of the International Conference on Learning Representations (ICLR), 2024. [45] Taoran Yi, Jiemin Fang, Guanjun Wu, Lingxi Xie, Xiaopeng Zhang, Wenyu Liu, Qi Tian, and Xinggang Wang. Gaussiandreamer: Fast generation from text to 3d gaussian splatting with point cloud priors. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2024. [46] Zihan Zhu, Songyou Peng, Viktor Larsson, Weiwei Xu, Hujun Bao, Zhaopeng Cui, Martin R. Oswald, and Marc Pollefeys. Nice-slam: Neural implicit scalable encoding for slam. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2022. [47] Zihan Zhu, Songyou Peng, Viktor Larsson, Zhaopeng Cui, Martin R Oswald, Andreas Geiger, and Marc Pollefeys. Nicer-slam: Neural implicit scene encoding for rgb slam. International Conference on 3D Vision (3DV), 2024. [48] M. Zwicker, H. Pfister, J. van Baar, and M. Gross. Ewa splatting. IEEE Transactions on Visualization and Computer Graphics, 8(3):223–238, 2002.
    """

    # Parse the references
    refs = parse_references(reference_text)

    # Your paragraph text
    paragraph = """
    Dense visual SLAM focuses on reconstructing detailed 3D maps, unlike sparse SLAM methods which excel in pose estimation [4, 5, 21] but typically yield maps useful mainly for localisation. In contrast, dense SLAM creates interactive maps beneficial for broader applications, including AR and robotics. Dense SLAM methods are generally divided into two primary categories: Frame-centric and Map-centric. Frame-centric SLAM minimises photometric error across consecutive frames, jointly estimating per-frame depth and frame-to-frame camera motion. Frame-centric approaches [1, 36] are efficient, as individual frames host local rather than global geometry (e.g. depth maps), and are attractive for long-session SLAM, but if a dense global map is needed, it must be constructed on demand by assembling all of these parts which are not necessarily fully consistent. In contrast, Map-centric SLAM uses a unified 3D representation across the SLAM pipeline, enabling a compact and streamlined system. Compared to purely local frame-to-frame tracking, a map-centric approach leverages global information by tracking against the reconstructed 3D consistent map. Classical map-centric approaches often use voxel grids [2, 23, 26, 40] or points [9, 29, 41] as the underlying 3D representation. While voxels enable a fast look-up of features in 3D, the representation is expensive, and the fixed voxel resolution and distribution are problematic when the spatial characteristics of the environment are not known in advance. On the other hand, a point-based map representation, such as surfel clouds, enables adaptive changes in resolution and spatial distribution by dynamic allocation of point primitives in the 3D space. Such flexibility benefits online applications such as SLAM with deformation-based loop closure [29, 41]. However, optimising the representation to capture high fidelity is challenging due to the lack of correlation among the primitives. Recently, in addition to classical graphic primitives, neural network-based map representations are a promising alternative. iMAP [33] demonstrated the interesting properties of neural representation, such as sensible hole filling of unobserved geometry. Many recent approaches combine the classical and neural representations to capture finer details [8, 27, 46, 47]; however, the large amount of computation required for neural rendering makes the live operation of such systems challenging.

    ## NeRF-based RGBD dense SLAM

    The classical method for creating a 3D representation was to unproject 2D observations into 3D space and to fuse them via weighted averaging [16, 23]. Such an averaging scheme suffers from over-smooth representation and lacks the expressiveness to capture high-quality details. To capture a scene with photorealistic quality, differentiable volumetric rendering [24] has recently been popularised with Neural Radiance Fields (NeRF) [17]. Using a single Multi-Layer Perceptron (MLP) as a scene representation, NeRF performs volume rendering by marching along pixel rays, querying the MLP for opacity and colour. Since volume rendering is naturally differentiable, the MLP representation is optimised to minimise the rendering loss using multiview information to achieve high-quality novel view synthesis. The main weakness of NeRF is its training speed. Recent developments have introduced explicit volume structures such as multi-resolution voxel grids [6, 14, 34] or hash functions [19] to improve performance. Interestingly, these projects demonstrate that the main contributor to high-quality novel view synthesis is not the neural network but rather differentiable volumetric rendering, and that it is possible to avoid the use of an MLP and yet achieve comparable rendering quality to NeRF [6]. However, even in these systems, per-pixel ray marching remains a significant bottleneck for rendering speed. This issue is particularly critical in SLAM, where immediate interaction with the map is essential for tracking. In contrast to NeRF

    ## Gaussian-based RGBD dense SLAM

    3DGS performs differentiable rasterisation. Similar to regular graphics rasterisations, by iterating over the primitives to be rasterised rather than marching along rays, 3DGS leverages the natural sparsity of a 3D scene and achieves a representation which is expressive to capture high-fidelity 3D scenes while offering significantly faster rendering. Several works have applied 3D Gaussians and differentiable rendering to static scene capture [11, 38], and in particular more recent works utilise 3DGS and demonstrate superior results in vision tasks such as dynamic scene capture [15, 42, 44] and 3D generation [35, 45]. Our method adopts a Map-centric approach, utilising 3D Gaussians as the only SLAM representation. Similar to surfel-based SLAM, we dynamically allocate the 3D Gaussians, enabling us to model an arbitrary spatial distribution in the scene. Unlike other methods such as ElasticFusion [41] and PointFusion [9], however, by using differentiable rasterisation, our SLAM system can capture highfidelity scene details and represent challenging object properties by direct optimisation against information from every pixel
    """

    # Replace citations in the paragraph
    result = replace_citations(paragraph, refs)
    print(result)
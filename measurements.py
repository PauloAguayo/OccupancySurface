import numpy as np

class Measurements(object):
    def __init__(self,gt_pol):
        self.gt_pol = gt_pol

    def iou(self,bb_test,bb_gt):

        def order_points(corner): # x,y
            f_corner = [0,0,0,0]
            if corner[1]<corner[3]:  # y_i < y_f
                f_corner[1], f_corner[3] = corner[1], corner[3]
            else:
                f_corner[1], f_corner[3] = corner[3], corner[1]
            if corner[0]<corner[2]: # x_i < x_f
                f_corner[0], f_corner[2] = corner[0], corner[2]
            else:
                f_corner[0], f_corner[2] = corner[2], corner[0]
            return(f_corner)

        bb_test = order_points(bb_test)
        xx1 = np.maximum(bb_test[0], bb_gt[0])
        yy1 = np.maximum(bb_test[1], bb_gt[1])
        xx2 = np.minimum(bb_test[2], bb_gt[2])
        yy2 = np.minimum(bb_test[3], bb_gt[3])
        w = np.maximum(0., xx2 - xx1)
        h = np.maximum(0., yy2 - yy1)
        wh = w * h
        o = wh / ((bb_test[2]-bb_test[0])*(bb_test[3]-bb_test[1])
          + (bb_gt[2]-bb_gt[0])*(bb_gt[3]-bb_gt[1]) - wh)
        return(o)


    def Area_Voronoi(self,hull_pol,hull_pol_mini):
        return(float(self.gt_pol*hull_pol_mini/hull_pol))

    def line_intersection(self,line1, line2):
        xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
        ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

        def det(a, b):
            return(a[0] * b[1] - a[1] * b[0])

        div = det(xdiff, ydiff)
        if div == 0:
           return([-100,-100])

        d = (det(*line1), det(*line2))
        x = det(d, xdiff) / div
        y = det(d, ydiff) / div
        return(np.array([round(x), round(y)]))


    def _2_voronoi(self,begin_end,polygon_added):
        polygones = [[],[]]
        corte = []
        activador = 0
        for en,ot in enumerate(polygon_added):
            for pt in np.array(begin_end):
                if (ot==pt).all():
                    activador+=1
                    corte.append(en)
                if 0<activador<2:
                    polygones[0].append(ot)
                elif activador==2:
                    polygones[0].append(ot)
                    activador+=1
                    break
        for ot in polygon_added[corte[1]:]:
            polygones[1].append(ot)
        for ot in polygon_added[:corte[0]+1]:
            polygones[1].append(ot)
        return(polygones)

    def n_voronoi(self, vor, centroid):
        new_regions = []
        new_vertices = vor.vertices.tolist()

        center = vor.points.mean(axis=0)
        radius = vor.points.ptp().max()*2

        # Construct a map containing all ridges for a given point
        all_ridges = {}
        for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
            all_ridges.setdefault(p1, []).append((p2, v1, v2))
            all_ridges.setdefault(p2, []).append((p1, v1, v2))

        for p1, region in enumerate(vor.point_region):
            vertices = vor.regions[region]

            # Finite region
            if all(v >= 0 for v in vertices):
                new_regions.append(vertices)
                continue

            # Non-finite region
            ridges = all_ridges[p1]
            new_region = [v for v in vertices if v >= 0]

            for p2, v1, v2 in ridges:
                if v2 < 0:
                    v1, v2 = v2, v1
                if v1 >= 0:
                    continue

                # Compute the missing endpoint of an infinite ridge
                t = vor.points[p2] - vor.points[p1] # tangent
                t /= np.linalg.norm(t)
                n = np.array([-t[1], t[0]])  # normal

                midpoint = vor.points[[p1, p2]].mean(axis=0)
                direction = np.sign(np.dot(midpoint - center, n)) * n*1700
                far_point = vor.vertices[v2] + direction * radius

                far_point[0] = far_point[0]-10000 if far_point[0]<centroid[0] else far_point[0]+10000
                far_point[1] = far_point[1]-10000 if far_point[1]<centroid[1] else far_point[1]+10000

                new_region.append(len(new_vertices))
                new_vertices.append(far_point.tolist())

            # sort region counterclockwise
            vs = np.asarray([new_vertices[v] for v in new_region])
            c = vs.mean(axis=0)
            angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
            new_region = np.array(new_region)[np.argsort(angles)]

            # finish
            new_regions.append(new_region.tolist())

        return([new_regions, np.asarray(new_vertices)])

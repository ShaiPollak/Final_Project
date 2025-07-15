import numpy as np
from scipy.interpolate import interp1d


class trajectory:
    def __init__(self, x, y, t, ids):
        # store trajectory data
        self.x = x
        self.y = y
        self.t = t
        self.ids = ids
        self.T = None
        self.x0 = None
        self.y0 = None
        self.xt = None
        self.yt = None

    @staticmethod
    def monotonic(x):
        """
        Test if an array is in a monotonic order
        Args:
            x: array

        Returns:
            True or False
        """
        # True if array is monotonic.
        # monotonic(X) returns True (False) if X is (not) monotonic.
        if len(x) == 1:
            return False

        if x[0] - x[1] > 0:
            x = np.flip(x, 0)

        dif = np.diff(x)
        if np.any(dif - np.abs(dif)):
            return False
        else:
            return True

    @staticmethod
    def trajectory_interpolation(t, x, y, s):
        """
        Interpolation function x(t), y(t) describe the locations at time t of a trajectory
        Args:
            t: time t of a trajectory
            x: longitude of the trajectory
            y: latitude of the trajectory
            s: oversampling coefficient (1: daily interpolation, 2: bidaily, 12: every 2h, etc.)

        Returns:
            ti: time of the interpolated trajectory
            fx(ti): longitude of the trajectory at the interpolated time
            fy(ti): latitude of the trajectory at the interpolated time
        """
        # interpolation functions
        fx = interp1d(t, x)
        fy = interp1d(t, y)

        # time where values will be interpolated
        days = np.floor(t[-1] - t[0])
        ti = np.linspace(t[0], t[0] + days, int(s * days + 1))
        return ti, fx(ti), fy(ti)

    
    def intersection_ratio(x1, x2):
        """
        Function used to interpolate trajectories at the ±180 meridian
        Args:
            x1: longitude on one side of the ±180 meridian
            x2: longitude on the other side of the ±180 meridian

        Returns:
            the ratio between x1 and ±180 meridian over x1 and x2
        """
        if x1 < 0:
            return (x1 + 180) / (360 - np.abs(x1 - x2))
        else:
            return (180 - x1) / (360 - np.abs(x1 - x2))

    def create_segments(self, T_days, mean_hours=1):
        """
        Subdivide trajectories into (x0, y0) -> (xt, yt) segments T_days apart.
        Automatically accounts for timestep resolution.
        """
        self.T = T_days

        T = int(T_days * 24 / mean_hours)  # number of time steps

        x0_list, y0_list, xt_list, yt_list = [], [], [], []

        I = np.where(np.diff(self.ids) != 0)[0]
        I = np.insert(I, [0, len(I)], [-1, len(self.ids) - 1])

        for j in range(len(I) - 1):
            start = I[j] + 1
            end = I[j + 1] + 1

            x = self.x[start:end]
            y = self.y[start:end]

            if len(x) <= abs(T):
                continue

            if T > 0:
                x0_list.append(x[:-T])
                y0_list.append(y[:-T])
                xt_list.append(x[T:])
                yt_list.append(y[T:])
            else:
                x0_list.append(x[-T:])
                y0_list.append(y[-T:])
                xt_list.append(x[:T])
                yt_list.append(y[:T])

        self.x0 = np.concatenate(x0_list)
        self.y0 = np.concatenate(y0_list)
        self.xt = np.concatenate(xt_list)
        self.yt = np.concatenate(yt_list)

    def filtering(self, x_range=None, y_range=None, t_range=None, complete_track=True):
        """
        Returns trajectory in a spatial domain and/or temporal range
        Args:
            x_range: longitudinal range (ascending order)
            y_range: meridional range (ascending order)
            t_range: temporal range (ascending order)
            complete_track: True: full trajectories is plotted
                            False: trajectories is plotted after it reaches the region

        Returns:
            segs [Ns, 2, 2]: list of segments to construct a LineCollection object
                - i: segment number (Ns)
                - j: coordinates of the beginning (0) or end (1) of the segment i
                - k: longitude (0) or latitude (1) of the coordiates j
            segs_t [Ns]: time associated to each segments
            segs_ind [Nt,2]: indices of the first and last segment of a trajectory
                     ex: trajectory 0 contains the segs[segs_ind[0,0]:segs_ind[0,1]]
        """
        if x_range is None:
            x_range = [np.min(self.x), np.max(self.x)]
        if y_range is None:
            y_range = [np.min(self.y), np.max(self.y)]
        if t_range is None:
            t_range = [np.min(self.t), np.max(self.t)]

        # identified drifters change
        I = np.where(abs(np.diff(self.ids, axis=0)) > 0)[0]
        I = np.insert(I, [0, len(I)], [-1, len(self.ids) - 1])

        segs = np.empty((0, 2, 2))
        segs_t = np.empty(0)
        segs_ind = np.zeros((0, 2), dtype="int")
        for j in range(0, len(I) - 1):
            range_j = np.arange(I[j] + 1, I[j + 1] + 1)
            xd = self.x[range_j]
            yd = self.y[range_j]
            td = self.t[range_j]

            # keep trajectory inside the specific domain and time frame
            keep = np.logical_and.reduce(
                (
                    xd >= x_range[0],
                    xd <= x_range[1],
                    yd >= y_range[0],
                    yd <= y_range[1],
                    td >= t_range[0],
                    td <= t_range[1],
                )
            )

            if np.sum(keep) > 1:
                # if complete_track: full trajectories is plotted
                # else: only after reaching the region
                if not complete_track:
                    reach = np.argmax(keep)
                    range_j = range_j[reach:]
                    xd = xd[reach:]
                    yd = yd[reach:]
                    td = td[reach:]

                if len(xd) > 1:
                    # search for trajectories crossing the ±180
                    cross_world = np.where(np.diff(np.sign(xd)))[0]
                    cross_world = np.unique(
                        np.insert(cross_world, [0, len(cross_world)], [-1, len(xd) - 1])
                    )

                    for k in range(0, len(cross_world) - 1):
                        ind = np.arange(cross_world[k] + 1, cross_world[k + 1] + 1)

                        if len(ind) > 1:
                            # reorganize array for LineCollection
                            pts = np.array([xd[ind], yd[ind]]).T.reshape(-1, 1, 2)
                            segs_i = np.concatenate([pts[:-1], pts[1:]], axis=1)

                            if len(segs_i) > 1:
                                segs_t_i = np.convolve(
                                    td, np.repeat(1.0, 2) / 2, "valid"
                                )  # average per segment
                            else:
                                segs_t_i = td

                            segs_ind = np.vstack(
                                (
                                    segs_ind,
                                    np.array([len(segs), len(segs) + len(segs_i)]),
                                )
                            )
                            segs = np.concatenate((segs, segs_i), axis=0)
                            segs_t = np.concatenate((segs_t, segs_t_i), axis=0)
        return segs, segs_t, segs_ind
package com.google.ar.core.helpers;

// https://stackoverflow.com/questions/49026297/convert-3d-world-arcore-anchor-pose-to-its-corresponding-2d-screen-coordinates

import android.opengl.Matrix;

import com.google.ar.core.Pose;

public class CoordinateTransformation {
    public static float[] calculateWorld2CameraMatrix(float[] modelmtx, float[] viewmtx, float[] prjmtx) {

        float scaleFactor = 1.0f;
        float[] scaleMatrix = new float[16];
        float[] modelXscale = new float[16];
        float[] viewXmodelXscale = new float[16];
        float[] world2screenMatrix = new float[16];

        Matrix.setIdentityM(scaleMatrix, 0);
        scaleMatrix[0] = scaleFactor;
        scaleMatrix[5] = scaleFactor;
        scaleMatrix[10] = scaleFactor;

        Matrix.multiplyMM(modelXscale, 0, modelmtx, 0, scaleMatrix, 0);
        Matrix.multiplyMM(viewXmodelXscale, 0, viewmtx, 0, modelXscale, 0);
        Matrix.multiplyMM(world2screenMatrix, 0, prjmtx, 0, viewXmodelXscale, 0);

        return world2screenMatrix;

    }

    public static double[] world2Screen(Pose pos, int screenWidth, int screenHeight, float[] world2cameraMatrix) {
        float[] origin = {0f, 0f, 0f, 1f};
        float[] ndcCoord = new float[4];
        Matrix.multiplyMV(ndcCoord, 0, world2cameraMatrix, 0, origin, 0);

        ndcCoord[0] = ndcCoord[0] / ndcCoord[3];
        ndcCoord[1] = ndcCoord[1] / ndcCoord[3];

        double[] pos_2d = new double[]{0, 0};
        pos_2d[0] = screenWidth * ((ndcCoord[0] + 1.0) / 2.0);
        pos_2d[1] = screenHeight * ((1.0 - ndcCoord[1]) / 2.0);

        return pos_2d;
    }

    public static double[] world2Screen(Pose pos, int screenWidth, int screenHeight, float[] viewmtx, float[] prjmtx) {
        float[] modelmtx = new float[16];
        pos.toMatrix(modelmtx, 0);
        return world2Screen(pos, screenWidth, screenHeight, calculateWorld2CameraMatrix(modelmtx, viewmtx, prjmtx));
    }
}
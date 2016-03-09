//		double[][] A = Utils.multiply(Utils.transpose(otherFeatureMatrix), otherFeatureMatrix);
//		double[][] AInverse = aMatrix.inverse().getArray();
//		
//		double[] muTemp = Utils.multiply(featureVector, AInverse); //inverse
//		muTemp = Utils.multiply(muTemp, Utils.transpose(otherFeatureMatrix));
//		mu = Utils.dot(muTemp, otherXVector);
//		xProbParameter[0] = mu;
		
//		double[] sigmaTemp = Utils.multiply(featureVector, AInverse); //inverse
//		sigma = Utils.dot(sigmaTemp, featureVector);
//		xProbParameter[1] = sigma;	
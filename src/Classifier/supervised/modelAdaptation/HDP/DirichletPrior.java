package Classifier.supervised.modelAdaptation.HDP;

import java.util.Arrays;

import cern.jet.random.Gamma;
/**
 * Dirichlet distribution, implemented by gamma function.
 * Referal: https://en.wikipedia.org/wiki/Dirichlet_distribution#Random_number_generation
 * @author lin
 */
public class DirichletPrior {
	
	//Sampling from parameters [\alpha_1, \alpha_2,...,\alpha_k].
	public void sampling(double[] target, double[] alphas){
//		target = new double[alphas.length];
		double sum = 0;
		for(int i=0; i<target.length; i++){
			target[i]= Gamma.staticNextDouble(alphas[i], 1);
			sum += target[i];
		}
		for(int i=0; i<target.length; i++) 
			target[i]/=sum;
	}
	
	//Sampling from parameters [\alpha/k, \alpha/k,..., \alpha/k]
	public void sampling(double[] target, double alpha, double dim){
		double[] alphas = new double[(int)dim];
		Arrays.fill(alphas, alpha/dim);
		sampling(target, alphas);
	}
	
//	public static void main(String[] args){
//		double[] a = new double[]{1, 2};
//		for(int i=0; i<a.length; i++) a[i]/=10;
//		for(double aa: a) System.out.println(aa);
//	}
}

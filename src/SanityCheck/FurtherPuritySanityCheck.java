package SanityCheck;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;

import structures.MyPriorityQueue;
import structures._Corpus;
import structures._Doc;
import structures._RankItem;
import structures._SparseFeature;
import utils.Utils;

public class FurtherPuritySanityCheck extends PuritySanityCheck {
	double[] m_BoWSimilarity;//Cache bow similarity.
	double[] m_TPSimilarity;//Cache topic similarity.
	ArrayList<CompareUnit> m_diffUnits;
	
	ArrayList<_Doc> m_documents;
	ArrayList<String> m_features;
	
	double[][] m_meanVar; //Save all units' similarity mean and variance.
	//The definition of compare unit.
	class CompareUnit implements Comparable<CompareUnit>{
		int n_index;
		int n_YLabel;
		int[] n_BoWIndex; //The indexes of all the neighbors returned by BoW.
		int[] n_TPIndex; //The indexes of all the neighbors returned by topic vector.
		int[] n_BoWLabels;
		int[] n_TPLabels;
		double n_BoWPurity;
		double n_TPPurity;
		double n_purityDiff;
		
		public CompareUnit(int index, int label, int[] BIs, int[] TIs, int[] BLs, int[] TLs){
			n_index = index;
			n_YLabel = label;
			n_BoWIndex = BIs;
			n_TPIndex = TIs;
			n_BoWLabels = BLs;
			n_TPLabels = TLs;
		}
		public void calculatePurity(){
			double[] count = new double[2];
			for(int i=0; i<n_BoWIndex.length; i++){
				if(n_BoWLabels[i] == n_YLabel)
					count[0]++;
				if(n_TPLabels[i] == n_YLabel)
					count[1]++;
			}
			n_BoWPurity = count[0]/n_BoWIndex.length;
			n_TPPurity = count[1]/n_TPIndex.length;
			n_purityDiff = n_BoWPurity - n_TPPurity; //The difference between the two purities returned.
		}
		//Judge if BoW purity is higher than TP purity.
//		public boolean BoWoverTP(){
//			if(n_BoWPurity > n_TPPurity)
//				return true;
//			else return false;
//		}
		
		@Override
		public int compareTo(CompareUnit o) {
			if(n_purityDiff < o.n_purityDiff)
				return 1;
			else if(n_purityDiff > o.n_purityDiff)
				return -1;
			return 0;
		}
		
//		public int compareTo(CompareUnit o) {
//			if(n_BoWPurity < o.n_BoWPurity)
//				return 1;
//			else if(n_BoWPurity > o.n_BoWPurity)
//				return -1;
//			return 0;
//		}
		
//		public int compareTo(CompareUnit o) {
//			if(n_TPPurity < o.n_TPPurity)
//				return 1;
//			else if(n_TPPurity > o.n_TPPurity)
//				return -1;
//			return 0;
//		}
	}
	
	public FurtherPuritySanityCheck(_Corpus c){
		super(c);
		m_diffUnits = new ArrayList<CompareUnit>();
		m_documents = c.getCollection();
	}
	
	public void setFeature(ArrayList<String> fs){
		m_features = fs;
	}
	//Calculate different similarities of two documents.
	public void calculateSimilarity(){
		_Doc di, dj;
		ArrayList<_Doc> documents = m_corpus.getCollection();
		m_BoWSimilarity = new double[m_size*(m_size-1)/2];
		m_TPSimilarity = new double[m_size*(m_size-1)/2];
		
		//Cache different similarities.
		for(int i=1; i<m_size; i++){
			for(int j=0; j<i; j++){
				di = documents.get(i);
				dj = documents.get(j);
				m_BoWSimilarity[getIndex(i, j)] = Utils.calculateSimilarity(di, dj);
				m_TPSimilarity[getIndex(i, j)] = Math.exp(-Utils.klDivergence(di.m_topics, dj.m_topics));
			}
		}
	}
	
	// Construct the units for comparison.
	public void constructCompareUnits(int k){
		_Doc tmpD;//the current document.
		int[] BoWIndex = new int[k];
		int[] TPIndex = new int[k];
		int[] BoWLabels = new int[k];
		int[] TPLabels = new int[k];
		_RankItem tmp;
		for(int i=0; i<m_size; i++){
			tmpD = m_corpus.getCollection().get(i);
			MyPriorityQueue<_RankItem> BoWneighborQueue= new MyPriorityQueue<_RankItem>(k);
			MyPriorityQueue<_RankItem> TPneighborQueue= new MyPriorityQueue<_RankItem>(k);
			//select top k most similar documents as neighbors.
			for(int j=0; j<m_size; j++){
				if(j==i)
					continue;
				else{
					BoWneighborQueue.add(new _RankItem(j, m_BoWSimilarity[getIndex(i, j)]));
					TPneighborQueue.add(new _RankItem(j, m_TPSimilarity[getIndex(i, j)]));
				}
			}
			//Traverse the top K neighbors to collect their labels from both BoW and topic similarity.
			for(int l=0; l<k; l++){
				tmp = BoWneighborQueue.get(l);
				BoWIndex[l] = tmp.m_index;
				BoWLabels[l] = m_documents.get(tmp.m_index).getYLabel();
			
				tmp = TPneighborQueue.get(l);
				TPIndex[l] = tmp.m_index;
				TPLabels[l] = m_documents.get(tmp.m_index).getYLabel();
			}
			CompareUnit unit = new CompareUnit(i, tmpD.getYLabel(), Arrays.copyOf(BoWIndex, k), Arrays.copyOf(TPIndex, k), Arrays.copyOf(BoWLabels, k), Arrays.copyOf(TPLabels, k));
			unit.calculatePurity();
			m_diffUnits.add(unit);
				
			//We don't need to clear, they will be replaced.
			Arrays.fill(BoWIndex, 0);
			Arrays.fill(TPIndex, 0);
			Arrays.fill(BoWLabels, 0);
			Arrays.fill(TPLabels, 0);
		}
		//Sort all the units based on the difference between two purities.
		Collections.sort(m_diffUnits);
	}
	
	//Print out the sorted purity difference.
	public void printDifference(String folder){
		try{
			m_meanVar = new double[m_diffUnits.size()][4];
			int BoWCount = 0, equalCount = 0;
			PrintWriter writer = new PrintWriter(new File(folder + "SortedBoWStat.xls"));
			
			// Get the number of units with larger BoW vs TP, equal BoW vs TP, smaller BoW vs TP.
			for(int i=0; i<m_diffUnits.size(); i++){
				if(m_diffUnits.get(i).n_purityDiff > 0)
					BoWCount++;
				else if(m_diffUnits.get(i).n_purityDiff == 0)
					equalCount++;
			}
			// Print out the units with larger BoW similarities.
			printOnePart(writer, 0, BoWCount, 0);
			System.out.format("%d reviews have better BoW purity.\n", BoWCount);
			
			printOnePart(writer, BoWCount, BoWCount + equalCount, 2);
			System.out.format("%d reviews have same BoW and TP purity.\n", equalCount);
			
			printOnePart(writer, BoWCount + equalCount, m_diffUnits.size(), 1); //TP is better than BoW, set flag = 1.
			System.out.format("%d reviews have better TP purity.\n", m_diffUnits.size() - equalCount - BoWCount);
			
			writer.close();
		}
		catch(IOException e){
			e.printStackTrace();
		}
	}
	
	// Write out results based on different purity difference.
	public void printOnePart(PrintWriter writer, int start, int end, int flag){
		CompareUnit unit;
		_Doc neighbor;
		int negCount = 0, posCount = 0;
//		double smeanSum = 0, svarSum = 0, lmeanSum = 0, lvarSum = 0;
	
		for(int i=start; i<end; i++){
			unit = m_diffUnits.get(i); //Access one unit.
			if(unit.n_YLabel == 0) 
				negCount++;
			else 
				posCount++;
			
			writer.write("==================================================\n");
			writer.format("trueL: %d\tDiff: %.4f\tBoW: %.4f\tTP: %.4f\t%s\n", unit.n_YLabel, unit.n_purityDiff, unit.n_BoWPurity, unit.n_TPPurity, m_documents.get(unit.n_index).getSource());
			writer.write("==================================================\n");
		
			//If TP > BoW, print TP first.
			if(flag == 1){
				for(int index: unit.n_TPIndex){
					neighbor = m_documents.get(index);
					writer.format("Label: %d\tTP Sim:%.4f\n", neighbor.getYLabel(), m_TPSimilarity[getIndex(unit.n_index, index)]);
					for(int j=0; j<neighbor.getTopics().length; j++)
						writer.format("(%d, %.4f)\t", j, neighbor.getTopics()[j]);
					writer.format("\n%s\n", neighbor.getSource());
				}
				writer.write("-----------------------------------------------\n");
				for(int index: unit.n_BoWIndex){
					neighbor = m_documents.get(index);
					writer.format("Label: %d\tBoW Sim:%.4f\n", neighbor.getYLabel(), m_BoWSimilarity[getIndex(unit.n_index, index)]);
					for(_SparseFeature sf: neighbor.getSparse())
						writer.format("(%s, %.4f)\t", m_features.get(sf.getIndex()), sf.getValue());
					writer.format("\n%s\n", neighbor.getSource());
				}
			//If BoW > TP, print BoW first.
			} else if(flag == 0){ 
				for(int index: unit.n_BoWIndex){
					neighbor = m_documents.get(index);
					writer.format("Label: %d\tBoW Sim:%.4f\n", neighbor.getYLabel(), m_BoWSimilarity[getIndex(unit.n_index, index)]);
					for(_SparseFeature sf: neighbor.getSparse())
						writer.format("(%s, %.4f)\t", m_features.get(sf.getIndex()), sf.getValue());
					writer.format("\n%s\n", neighbor.getSource());
				}
				writer.write("-----------------------------------------------\n");
				for(int index: unit.n_TPIndex){
					neighbor = m_documents.get(index);
					writer.format("Label: %d\tTP Sim:%.4f\n", neighbor.getYLabel(), m_TPSimilarity[getIndex(unit.n_index, index)]);
					for(int j=0; j<neighbor.getTopics().length; j++)
						writer.format("(%d, %.4f)\t", j, neighbor.getTopics()[j]);
					writer.format("\n%s\n", neighbor.getSource());
				}
			}
			
//			m_meanVar[i][0] = smeanSum/indexes.length;//Calculate the mean of similarity.
//			m_meanVar[i][2] = lmeanSum/indexes.length;
			//Calculate the variance of similarities.
//			for(int index: indexes){
//				svarSum += (similarity[getIndex(unit.n_index, index)] - m_meanVar[i][0])*(similarity[getIndex(unit.n_index, index)] - m_meanVar[i][0]);
//				lvarSum += (m_documents.get(index).getDocLength() - m_meanVar[i][2])*(m_documents.get(index).getDocLength() - m_meanVar[i][2]);
//			}
//			m_meanVar[i][1] = svarSum/indexes.length;
//			m_meanVar[i][3] = lvarSum/indexes.length;
			
			//clear the two sums.
//			smeanSum = 0; svarSum = 0;
//			lmeanSum = 0; lvarSum = 0;
			writer.format("total\t%d\tpos\t%.4f\tneg\t%.4f\n", (posCount + negCount), (double) posCount/(posCount+negCount), (double) negCount/(posCount+negCount));
			writer.write("-----------------------------------------------\n");
		}
	}
	public void printBoWSimilarity(String folder){
		try{
			m_meanVar = new double[m_diffUnits.size()][2];
			double mSum = 0, vSum = 0;
			PrintWriter writer = new PrintWriter(new File(folder + "SortedBoW.xls"));
			CompareUnit unit;
			_Doc neighbor;
			int negCount = 0, posCount = 0;
			for(int i=0; i<m_diffUnits.size(); i++){
				unit = m_diffUnits.get(i); //Access one unit.
				if(unit.n_YLabel == 0) 
					negCount++;
				else 
					posCount++;
				writer.format("%d\t%.4f\t%.4f\t%s\n", unit.n_YLabel, unit.n_BoWPurity, unit.n_TPPurity, m_documents.get(unit.n_index).getSource());
				for(int in: unit.n_BoWIndex){
					neighbor = m_documents.get(in);
//					System.out.println(m_BoWSimilarity[getIndex(unit.n_index, in)]);
					mSum += m_BoWSimilarity[getIndex(unit.n_index, in)];
					writer.format("Label\t%d\tSim\t%.4f\n", neighbor.getYLabel(), m_BoWSimilarity[getIndex(unit.n_index, in)]);
					for(_SparseFeature sf: neighbor.getSparse()){
						writer.format("(%s, %.4f)\t", m_features.get(sf.getIndex()), sf.getValue());
					}
					writer.format("\n%s\n", neighbor.getSource());
				}
				m_meanVar[i][0] = mSum/(unit.n_BoWIndex.length);
				for(int in: unit.n_BoWIndex){
					vSum += (m_BoWSimilarity[getIndex(unit.n_index, in)] - m_meanVar[i][0])*(m_BoWSimilarity[getIndex(unit.n_index, in)] - m_meanVar[i][0]);
				}
				m_meanVar[i][1] = vSum/(unit.n_BoWIndex.length);
				mSum = 0; vSum = 0;
//				System.out.print("-----------------------------------------------\n");
				writer.write("-----------------------------------------------\n");
			}
			writer.format("neg\t%d\tpos\t%d\ttotal\t%d\n", negCount, posCount, posCount+negCount);
			writer.close();
		} catch(IOException e){
			e.printStackTrace();
		}
	}
	
	public void printTPSimilarity(String folder){
		try{
			m_meanVar = new double[m_diffUnits.size()][2];
			double mSum = 0, vSum = 0;
			PrintWriter writer = new PrintWriter(new File(folder + "SortedTP.xls"));
			CompareUnit unit;
			_Doc neighbor;
			int negCount = 0, posCount = 0;
			for(int i=0; i<m_diffUnits.size(); i++){
				unit = m_diffUnits.get(i); //Access one unit.
				if(unit.n_YLabel == 0) 
					negCount++;
				else 
					posCount++;
				writer.format("%d\t%.4f\t%.4f\t%s\n", unit.n_YLabel, unit.n_BoWPurity, unit.n_TPPurity, m_documents.get(unit.n_index).getSource());
				for(int in: unit.n_TPIndex){
					neighbor = m_documents.get(in);
					mSum += m_TPSimilarity[getIndex(unit.n_index, in)];
					writer.format("Label\t%d\tSim\t%.4f\n", neighbor.getYLabel(), m_TPSimilarity[getIndex(unit.n_index, in)]);
					for(_SparseFeature sf: neighbor.getSparse()){
						writer.format("(%s, %.4f)\t", m_features.get(sf.getIndex()), sf.getValue());
					}
					writer.format("\n%s\n", neighbor.getSource());
				}
				m_meanVar[i][0] = mSum/(unit.n_TPIndex.length);
				for(int in: unit.n_TPIndex){
					vSum += (m_TPSimilarity[getIndex(unit.n_index, in)] - m_meanVar[i][0])*(m_TPSimilarity[getIndex(unit.n_index, in)] - m_meanVar[i][0]);
				}
				m_meanVar[i][1] = vSum/(unit.n_TPIndex.length);
				mSum = 0; vSum = 0;
				writer.write("-----------------------------------------------\n");
			}
			writer.format("neg\t%d\tpos\t%d\ttotal\t%d\n", negCount, posCount, posCount+negCount);
			writer.close();
		} catch(IOException e){
			e.printStackTrace();
		}
	}
	
	//Print out the mean and var for each reviews with respect to neighbors.
	public void printMeanVar(String folder, String filename){
		try{
			PrintWriter writer = new PrintWriter(new File(folder + filename));
			for(int i=0; i<m_meanVar.length; i++)
				writer.format("%d\t%.4f\t%.4f\n", i, m_meanVar[i][0], m_meanVar[i][1]);
			writer.close();
		} catch(IOException e){
			e.printStackTrace();
		}
	}
	
	//Get the basic stat of the documents.
	public int[] getMinMaxDocLength(){
		int min = 1000, max = 0;
		_Doc tmp;
		for(int i=0; i<m_documents.size(); i++){
			tmp = m_documents.get(i);
			if(tmp.getDocLength() < min)
				min = tmp.getDocLength();
			if(tmp.getDocLength() > max)
				max = tmp.getDocLength();
		}
		return new int[]{min, max};
	}
	//Get the statistics of the document length and distribution.
	public void calcDocLengthStat(int max, int itv){
		int lenDim = max/itv + 1; //The number of doc sections. 15: [0, 10), [10, 20), [20, 30), [30, 40)
		double[][] stat = new double[lenDim][4]; // Each row is the len sections, each column is for class 0 and 1.
		_Doc tmp;
		int yLabel = 0;
		for(int i=0; i<m_documents.size(); i++){
			tmp = m_documents.get(i);
			yLabel = tmp.getYLabel();
			stat[tmp.getDocLength()/itv][yLabel]++;
		}
		//Print out the index and calculate the ratioes.
		for(int i=0; i<lenDim; i++){
			System.out.format("%d\t", i);
			stat[i][2] = stat[i][0]/(stat[i][0]+stat[i][1]+0.000001);//The ratio for negative docs.
			stat[i][3] = stat[i][1]/(stat[i][0]+stat[i][1]+0.000001);//The ratio for positive docs.
		}
		System.out.println();
		
		//Print out the neg ratio=neg/(neg+pos);
		for(int i=0; i<lenDim; i++)
			System.out.format("%.4f\t", stat[i][2]);
		System.out.println();
		
		//Print out the pos ratio=pos/(neg+pos);
		for(int i=0; i<lenDim; i++)
			System.out.format("%.4f\t", stat[i][3]);
		System.out.println();
		
		//Print out the absolute value of the pos docs and neg docs.
		for(int i=0; i<lenDim; i++)
			System.out.format("%d&%d\t", (int)stat[i][0], (int)stat[i][1]);
		System.out.println();
	}
}

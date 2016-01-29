package SanityCheck;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import structures.MyPriorityQueue;
import structures._Corpus;
import structures._Doc;
import structures._RankItem;
import utils.Utils;

public class BaseSanityCheck {
	
	// Different types of similarity calculation.
	public enum SimType {
		ST_BoW,  // BoW similarity
		ST_TP,   // Topic similarity
		ST_STP,	 // Sentiment topic similarity
		ST_Rand, // Random similarity
		ST_L2R	 // L2R similarity
	}
	
	int m_size; // the total number of documents.
	double[] m_similarity;
	double[][] m_avgPatK;//m_PatK[0] is for negative class and m_PatK[1] is for positive class.
	double[][] m_PatK; //PatK for each document.
	double[] m_counts;
	
	SimType m_sType;
	_Corpus m_corpus;
	ArrayList<_Doc> m_documents;
	ArrayList<String> m_features;
	MyPriorityQueue<_RankItem> m_neighborQueue;
	
	public BaseSanityCheck(_Corpus c){
		m_sType = SimType.ST_Rand; //Specify random as default.
		m_corpus = c;
		m_size = c.getCollection().size();
		m_counts = new double[2];
		m_documents = m_corpus.getCollection();
		m_neighborQueue = new MyPriorityQueue<_RankItem>(20);
	}
	
	public BaseSanityCheck(_Corpus c, SimType sType){
		m_sType = sType; //"BoW", "TP", "L2R"
		m_corpus = c;
		m_size = c.getCollection().size();
		m_counts = new double[2];
		m_documents = m_corpus.getCollection();
	}
	
	//Calculate the similarity in advance based on the selection method.
	public void calculateSimilarity(){
		_Doc di, dj;
		ArrayList<_Doc> documents = m_corpus.getCollection();
		m_similarity = new double[m_size*(m_size-1)/2];
		
		for(int i=1; i<m_size; i++){
			for(int j=0; j<i; j++){
				di = documents.get(i);
				dj = documents.get(j);
				m_similarity[getIndex(i, j)] = calculateSimilarity(di, dj, m_sType);
			}
		}
	}
	
	// Calcualte similarity based on different calculation methods.
	public double calculateSimilarity(_Doc di, _Doc dj, SimType sType){
		if(m_sType == SimType.ST_Rand)
			return Math.random();
		if(m_sType == SimType.ST_BoW)
			return Utils.calculateSimilarity(di, dj);
		else if(m_sType == SimType.ST_TP)
			return Utils.klDivergence(di.getTopics(), dj.getTopics());
		else if(m_sType == SimType.ST_L2R)
			return 0;
		else
			return 0;
	}
	
	public int getIndex(int i, int j){
		//Swap i and j.
		if(i < j){
			int t = j;
			j = i;
			i = t;
		}
		return i*(i-1)/2+j;
	}
	
	//Calculate p@5, p@10, p@20
	public void calculatePatK4All(int topK){
		if(m_neighborQueue == null || m_neighborQueue.size() == 0 )
			m_neighborQueue = new MyPriorityQueue<_RankItem>(topK);

		_Doc doc;//the current document.
		int[] neighbors = new int[topK];		
		m_avgPatK = new double[2][3]; // p@5, p@10, p@20
		m_PatK = new double[m_size][3];
		
		for(int i=0; i<m_size; i++){
			doc = m_documents.get(i);
			//If we use random similarity.
			if(m_sType == SimType.ST_BoW){
				for(int j=0; j<20; j++){
					_Doc neighbor = m_documents.get((int)(Math.random() * m_size));
					neighbors[j] = neighbor.getYLabel();
				}
			} else{//else select neighbors based on similarity.
				updateNeighborQueue(doc, i);
				//Traverse the top K neighbors to collect their labels.
				for(int k=0; k<m_neighborQueue.size(); k++){
					_RankItem tmp = m_neighborQueue.get(k);
					neighbors[k] = m_documents.get(tmp.m_index).getYLabel();
				}
				m_PatK[i][0] = calcPatK(neighbors, 5, doc.getYLabel());
				m_PatK[i][1] = calcPatK(neighbors, 10, doc.getYLabel());
				m_PatK[i][2] = calcPatK(neighbors, 20, doc.getYLabel());

				m_avgPatK[doc.getYLabel()][0] += m_PatK[i][0];
				m_avgPatK[doc.getYLabel()][1] += m_PatK[i][1];
				m_avgPatK[doc.getYLabel()][2] += m_PatK[i][2];
							
				m_counts[doc.getYLabel()]++;
			}
		}
	}
	
	public void writePatK(String filename){
		try{
			PrintWriter writer = new PrintWriter(new File(filename));
			writer.format("Inlink\tp@5\tp@10\tp@20\n");
			for(int i=0; i<m_size; i++)
				writer.format("%d\t%.4f\t%.4f\t%.4f\n", m_documents.get(i).getInlink(), m_PatK[i][0], m_PatK[i][1], m_PatK[i][2]);

			for(int i=0; i<2; i++){
				m_avgPatK[i][0] /= m_counts[i];
				m_avgPatK[i][1] /= m_counts[i];
				m_avgPatK[i][2] /= m_counts[i];
				writer.format("\t%.4f\t%.4f\t%.4f\n", m_avgPatK[i][0], m_PatK[i][1], m_PatK[i][2]);
			}			
			writer.close();
		} catch(IOException e){
			e.printStackTrace();
		}
	}
	
	//Construct the neighbor queue.
	public void updateNeighborQueue(_Doc d, int i) {
		m_neighborQueue.clear();
		for(int j=0; j<m_size; j++){
			if(j==i)
				continue;
			m_neighborQueue.add(new _RankItem(j, m_similarity[getIndex(i, j)]));
		}
	}
	
	public void calculateInlinks(int topK){
		if(m_neighborQueue == null || m_neighborQueue.size() == 0 )
			m_neighborQueue = new MyPriorityQueue<_RankItem>(topK);
		_Doc doc; 		
		for(int i=0; i<m_size; i++){
			doc = m_documents.get(i);
			updateNeighborQueue(doc, i);
			for(int k=0; k<m_neighborQueue.size(); k++){
				_RankItem item = m_neighborQueue.get(k);
				m_documents.get(item.m_index).addOneInlink();
			}
		}
		//Sort all the documents based on the inlink
		Collections.sort(m_documents, new Comparator<_Doc>(){
			public int compare(_Doc d1, _Doc d2){
				if(d1.getInlink() < d2.getInlink())
					return 1;
				else if(d1.getInlink() > d2.getInlink())
					return -1;
				else 
					return 0;
			}
		});
	}

	//Print p@5, p@10, p@20
	public void printPatK(){
		System.out.format("\tp@5\tp@10\tp@20\n");
		//Print out neg p@k.
		System.out.print("neg:");
		for(int i=0; i<3; i++)
			System.out.format("\t%.4f", m_PatK[0][i]);
		System.out.println();
		//Print out pos p@k.
		System.out.print("pos:");
		for(int i=0; i<3; i++)
			System.out.format("\t%.4f", m_PatK[1][i]);
		System.out.println();
	}
	public void printPatK(int topK, int itv){
		int interval = topK/itv;
		for(int i=0; i<interval; i++)
			System.out.format("\tp@%d", (i+1)*itv);
		System.out.println();
		//Print out neg p@k.
		for(int i=0; i<interval; i++)
			System.out.format("\t%.4f", m_PatK[0][i]);
		System.out.println();
		//Print out pos p@k.
		for(int i=0; i<interval; i++)
			System.out.format("\t%.4f", m_PatK[1][i]);
		System.out.println();
		
	}
	
	//calculate p@k for one document.
	public double calcPatK(int[] labels, int k, int label){
		if(labels.length < k)
			return 0;
		int count = 0;
		for(int i=0; i<k; i++){
			if(labels[i] == label)
				count++;
		}
		return (double)count/k;
	}
}

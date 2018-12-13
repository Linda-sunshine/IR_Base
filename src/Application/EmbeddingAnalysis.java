package Application;

import structures.MyPriorityQueue;
import structures._RankItem;
import utils.Utils;

import java.io.*;
import java.util.ArrayList;

public class EmbeddingAnalysis {

    int m_numOfTopics;
    int m_dim;
    double[][] m_topicEmbeddingMu;
    double[][][] m_topicEmbeddingSigma;


    public EmbeddingAnalysis(int nuTopics, int dim){
        m_numOfTopics = nuTopics;
        m_dim = dim;

        m_topicEmbeddingMu = new double[nuTopics][dim];
        m_topicEmbeddingSigma = new double[nuTopics][dim][dim];
    }
    public void loadTopicEmbeddings(String filename){
        try {
            File file = new File(filename);
            BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(file), "UTF-8"));
            int index = 0;
            String line, muStr, sigmaStr;

            while ((line = reader.readLine()) != null) {
                if(line.startsWith("Topic")){
                    index = Integer.valueOf(line.substring(6));
                    reader.readLine();// "mu_phi"
                    muStr = reader.readLine();
                    m_topicEmbeddingMu[index] = processOneLine(muStr);
                    reader.readLine();// "sigma_phi"
                    double[][] sigma = new double[m_dim][m_dim];
                    for(int i=0; i<m_dim; i++){
                        sigma[i] = processOneLine(reader.readLine());
                    }
                    m_topicEmbeddingSigma[index] = sigma;
                    reader.readLine();// "-------"
                } else
                    System.out.println("Error in reading embedding file!");
            }
            System.out.format("Finish loading %d topic embeddings from %s.\n", index+1, filename);
        }catch(IOException e){
            e.printStackTrace();
        }
    }

    public double[] processOneLine(String str){
        String[] strs = str.split("\t");
        double[] vals = new double[strs.length];
        for(int i=0; i<strs.length; i++){
            vals[i] = Double.valueOf(strs[i]);
        }
        return vals;
    }


    public void rankTopics(){
        double[][] m_sim = new double[m_numOfTopics][m_numOfTopics];
        for(int i = 0; i<m_numOfTopics; i++){
            for(int j = i+1; j<m_numOfTopics; j++){
                m_sim[i][j] = Utils.cosine(m_topicEmbeddingMu[i], m_topicEmbeddingMu[j]);
                m_sim[j][i] = m_sim[i][j];
            }
        }
        // rank topics for each topic embedding
        for(int i=0; i<m_numOfTopics; i++){
            MyPriorityQueue<_RankItem> q = new MyPriorityQueue<_RankItem>(10);
            for(int j=0; j<m_numOfTopics; j++){
                if(j == i) continue;
                q.add(new _RankItem(j, m_sim[i][j]));
            }
            System.out.format("Topic %d: ", i);
            for(_RankItem it: q) {
                System.out.format("(topic %d\t%.3f)\t\t", it.m_index, it.m_value);
            }
            System.out.println();
        }
        System.out.println("Finish ranking all topics!");
    }

    public static void main(String[] args) {
        int nuTopics = 30, dim = 10;
        EmbeddingAnalysis analysis = new EmbeddingAnalysis(nuTopics, dim);
        analysis.loadTopicEmbeddings("./data/TopicEmbedding.txt");
        analysis.rankTopics();

    }

}

package Application;

import structures.MyPriorityQueue;
import structures._RankItem;
import utils.Utils;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;

public class EmbeddingAnalysis {

    int m_numOfTopics;
    int m_dim;

    double[][] m_betas;
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

    // load the learned topics for comparison
    public void loadBeta(String filename){
        try {
            File file = new File(filename);
            BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(file), "UTF-8"));
            int index = 0;
            String line, muStr, sigmaStr;

            // the first line is about dimension: topic size * vocabulary size
            line = reader.readLine();
            String[] strs = line.split("\\s+");
            int topicSize = Integer.valueOf(strs[0]);
            if(topicSize != m_numOfTopics){
                System.err.println("Wrong topic size! Return");
                return;
            }
            int vocabSize = Integer.valueOf(strs[1]);
            m_betas = new double[m_numOfTopics][vocabSize];

            while ((line = reader.readLine()) != null) {
                m_betas[index++] = processOneLine(line);
            }
            System.out.format("Finish loading %d topics(betas) from %s.\n", index, filename);
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

    public void rankTopicEmbeddings(){
        double[][] m_sim = new double[m_numOfTopics][m_numOfTopics];
        for(int i = 0; i<m_numOfTopics; i++){
            for(int j = i+1; j<m_numOfTopics; j++){
                m_sim[i][j] = Utils.cosine(m_topicEmbeddingMu[i], m_topicEmbeddingMu[j]);
                m_sim[j][i] = m_sim[i][j];
            }
        }
        for(int i=0; i<m_numOfTopics; i++){

            MyPriorityQueue<_RankItem> q = new MyPriorityQueue<_RankItem>(m_numOfTopics-1);
            for(int j=0; j<m_numOfTopics; j++){
                if(j == i) continue;
                q.add(new _RankItem(j, m_sim[i][j]));
            }
            for(_RankItem it: q) {
                System.out.format("(topic %d\t%.3f)\t\t", it.m_index, it.m_value);
            }
            System.out.println();
        }
        System.out.println("Finish ranking all topics!");
    }

    double[][] m_embeddingSim, m_betaSim;
    public void calcTauCoef(){
        m_embeddingSim = new double[m_numOfTopics][m_numOfTopics];
        m_betaSim = new double[m_numOfTopics][m_numOfTopics];

        for(int i = 0; i<m_numOfTopics; i++){
            for(int j = i+1; j<m_numOfTopics; j++){
                m_embeddingSim[i][j] = Utils.cosine(m_topicEmbeddingMu[i], m_topicEmbeddingMu[j]);
                m_embeddingSim[j][i] = m_embeddingSim[i][j];

                m_betaSim[i][j] = Utils.cosine(m_betas[i], m_betas[j]);
                m_betaSim[j][i] = m_betaSim[i][j];
            }
        }
        // for each topic, calculate the tau coefficient
        double[] taus = new double[m_numOfTopics];
        for(int i=0; i<m_numOfTopics; i++){
            taus[i] = calcTau4OneTopic(i);
        }
        Arrays.sort(taus);
        for(double t: taus){
            System.out.format("%.2f\t", t);
        }
        System.out.println("Finish ranking all topics!");
    }

    public double calcTau4OneTopic(int i){

        MyPriorityQueue<_RankItem> embeddingQ = new MyPriorityQueue<_RankItem>(m_numOfTopics-1);
        MyPriorityQueue<_RankItem> betaQ = new MyPriorityQueue<_RankItem>(m_numOfTopics-1);

        for(int j=0; j<m_numOfTopics; j++){
            if(j == i) continue;
            embeddingQ.add(new _RankItem(j, m_embeddingSim[i][j]));
            betaQ.add(new _RankItem(j, m_betaSim[i][j]));
        }

        HashSet<String> embeddingTable = constructLookup(embeddingQ);
        HashSet<String> betaTable = constructLookup(betaQ);
        // concordant means the order of two ranking is the same
        // discordant means the order of two ranking is differnet
        double concordant = 0;
        for(String pair: embeddingTable){
            if(betaTable.contains(pair))
                concordant++;
        }
        return (concordant - embeddingTable.size())/(double) embeddingTable.size();
    }

    public HashSet<String> constructLookup(MyPriorityQueue<_RankItem> q){
        int[] items = new int[q.size()];
        int index = 0;
        for(_RankItem it: q){
            items[index++] = it.m_index;
        }
        HashSet<String> table = new HashSet<String>();
        for(int m=0; m<items.length; m++){
            for(int n=m+1; n<items.length; n++){
                table.add(String.format("%d-%d", items[m], items[n]));
            }
        }
        return table;
    }


    // rank topics for each topic embedding
    HashMap<String, Integer> m_lookup = new HashMap<String, Integer>();

    public static void main(String[] args) {
        int nuTopics = 50, dim = 10;
        EmbeddingAnalysis analysis = new EmbeddingAnalysis(nuTopics, dim);
        analysis.loadTopicEmbeddings("./data/yelp_fold_0/TopicEmbedding.txt");
        analysis.loadBeta("./data/yelp_fold_0/Beta.txt");
        analysis.calcTauCoef();

    }

}

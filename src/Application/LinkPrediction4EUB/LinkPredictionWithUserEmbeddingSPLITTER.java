package Application.LinkPrediction4EUB;

import utils.Utils;

import java.io.*;
import java.util.ArrayList;
import java.util.HashMap;

public class LinkPredictionWithUserEmbeddingSPLITTER extends LinkPredictionWithUserEmbedding {

    public LinkPredictionWithUserEmbeddingSPLITTER(){
        super();
    }

    HashMap<Integer, ArrayList<double[]>> m_multipleEmbeddings = new HashMap<>();

    // load each user's embedding
    @Override
    // load each user's embedding
    public void loadUserEmbedding(String filename) {
        try {
            File file = new File(filename);
            BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(file), "UTF-8"));
            String line;

            String firstLine = reader.readLine();
            String[] strs = firstLine.trim().split("\\s+");
            if (strs.length != 2) {
                System.out.println("[error]The file is not correct!! Double check user embedding file!");
                return;
            }
            m_dim = Integer.valueOf(strs[1]);
            // read each user's embedding one by one
            int count = 0;
            while ((line = reader.readLine()) != null) {
                String[] valStrs = line.trim().split("\\s+");
                if (valStrs.length != m_dim + 1) {
                    System.out.println("[error]The user's dimension is not correct!!");
                    continue;
                }
                String uid = valStrs[0];
                double[] embedding = new double[m_dim];
                for (int i = 1; i < valStrs.length; i++) {
                    embedding[i - 1] = Double.valueOf(valStrs[i]);
                }
                if(!m_idIndexMap.containsKey(uid))
                    continue;
                int index = m_idIndexMap.get(uid);
                if(!m_multipleEmbeddings.containsKey(index)){
                    m_multipleEmbeddings.put(index, new ArrayList<>());
                }
                m_multipleEmbeddings.get(index).add(embedding);
                count++;
            }

            int min = 10, max = 0;
            for(int idx: m_multipleEmbeddings.keySet()){
                int tmp = m_multipleEmbeddings.get(idx).size();
                if(tmp > max) max = tmp;
                if(tmp < min) min = tmp;
            }
            reader.close();
            System.out.format("[Info]Finish loading %d persona embeddings for %d user from %s.\n", count, m_multipleEmbeddings.size(), filename);
            System.out.format("[Info]Min number of roles: %d, max number of roles: %d\n", min, max);

        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void calcSimilarity(){
        System.out.println("Start calculating similarity based on personas....");

        m_similarity = new double[m_userSize][m_userSize];
        for(int i=0; i<m_userSize; i++){
            for(int j=i+1; j<m_userSize; j++){
                m_similarity[i][j] = calcMaxSim(i, j);
            }
        }
    }

    public double calcMaxSim(int i, int j){
        if(!m_multipleEmbeddings.containsKey(i) || !m_multipleEmbeddings.containsKey(j) )
            return 0;
        ArrayList<double[]> personaI = m_multipleEmbeddings.get(i);
        ArrayList<double[]> personaJ = m_multipleEmbeddings.get(j);
        double max = 0;
        for(int m=0; m<personaI.size(); m++){
            for(int n=0; n<personaJ.size(); n++){
                double tmp = Utils.dotProduct(personaI.get(m), personaJ.get(n));
                if(tmp > max)
                    max = tmp;
            }
        }
        return max;
    }
}

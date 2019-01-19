package Application.LinkPrediction4EUB;

import structures._SparseFeature;
import utils.Utils;

import java.io.*;
import java.util.ArrayList;
import java.util.HashMap;

public class LinkPredictionWithUserEmbeddingBOW extends LinkPredictionWithUserEmbedding {
    HashMap<Integer, ArrayList<_SparseFeature>> m_bows;
    public LinkPredictionWithUserEmbeddingBOW(){
        super();
    }

    // load each user's embedding
    @Override
    public void loadUserEmbedding(String filename){
        try {
            File file = new File(filename);
            BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(file), "UTF-8"));
            String line;

            String firstLine = reader.readLine();
            String[] strs = firstLine.trim().split("\t");
            if(strs.length != 2){
                System.out.println("[error]The file is not correct!! Double check user embedding file!");
                return;
            }
            int userSize = Integer.valueOf(strs[0]);
            m_dim = Integer.valueOf(strs[1]);

            m_bows = new HashMap<>();
            if(userSize > m_userIds.size()) {
                System.out.println("[error]The file is not correct!! Double check user embedding file!");
                return;
            }
            // read each user's embedding one by one
            int count = 0;
            while ((line = reader.readLine()) != null) {
                if(count > m_userSize){
                    System.out.println("[error]The line number exceeds the user size!!");
                    break;
                }
                String[] valStrs = line.trim().split("\t");
                if(valStrs.length % 2 != 1)
                    System.out.println("BOW dim is not correct!");
                String uid = valStrs[0];
                ArrayList<_SparseFeature> fvs = new ArrayList<>();
                for(int i=1; i<valStrs.length; i+=2){
                    _SparseFeature fv = new _SparseFeature(Integer.valueOf(valStrs[i]), Double.valueOf(valStrs[i+1]));
                    fvs.add(fv);
                }

                int index = m_idIndexMap.get(uid);
                m_bows.put(index, fvs);
                count++;
            }
            reader.close();
            System.out.format("[Info]Finish loading %d user embeddings from %s.\n", count, filename);
        } catch(IOException e){
            e.printStackTrace();
        }
    }

    public void calcSimilarity(){
        System.out.println("Start calculating similarity based on BOW....");

        m_similarity = new double[m_userSize][m_userSize];
        for(int i=0; i<m_userSize; i++){
            for(int j=i+1; j<m_userSize; j++){
                m_similarity[i][j] = Utils.cosine(convert(m_bows.get(i)), convert(m_bows.get(j)));
            }
        }
        System.out.println("Finish calculating similarity based on BOW.");
    }

    // convert arraylist to array
    public _SparseFeature[] convert(ArrayList<_SparseFeature> arr){
        _SparseFeature[] fvs = new _SparseFeature[arr.size()];
        for(int i=0; i<arr.size(); i++){
            fvs[i] = arr.get(i);
        }
        return fvs;
    }
}

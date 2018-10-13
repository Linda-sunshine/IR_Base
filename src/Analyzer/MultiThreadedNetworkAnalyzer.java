package Analyzer;

import opennlp.tools.util.InvalidFormatException;
import structures._Doc;
import structures._Review;
import structures._SparseFeature;
import structures._User;
import utils.Utils;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;

/**
 * @author Lin Gong (lg5bt@virginia.edu)
 * The analyzer aims at the directional network information.
 */
public class MultiThreadedNetworkAnalyzer extends MultiThreadedLinkPredAnalyzer {

    HashMap<String, HashSet<String>> m_networkMap = new HashMap<>();

    public MultiThreadedNetworkAnalyzer(String tokenModel, int classNo,
                                        String providedCV, int Ngram, int threshold, int numberOfCores, boolean b)
            throws IOException {
        super(tokenModel, classNo, providedCV, Ngram, threshold, numberOfCores, b);
    }

    // save one file for indexing users in later use (construct network for baselines)
    public void saveUserIds(String filename){
        try {
            PrintWriter writer = new PrintWriter(new File(filename));
            for(int i=0; i<m_users.size(); i++){
                writer.format("%s\t%d\n", m_users.get(i).getUserID(), i);
            }
            writer.close();
            System.out.format("Finish saving %d user ids.\n", m_users.size());
        } catch(IOException e){
            e.printStackTrace();
        }
    }
    // save the network for later use
    public void saveNetwork(String filename){
        try {
            PrintWriter writer = new PrintWriter(new File(filename));
            for(String uid: m_networkMap.keySet()){
                writer.write(uid + '\t');
                for(String it: m_networkMap.get(uid)){
                    writer.write(it + '\t');
                }
                writer.write('\n');
            }
            writer.close();
        } catch(IOException e){
            e.printStackTrace();
        }
    }

    // load the interactions, filter the users who are not in the user
    public void loadInteractions(String filename){
        try{
            File file = new File(filename);
            BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(file), "UTF-8"));
            String line;

            int count = 0;
            double avgFriendSize = 0;

            // load the interactions first
            while((line = reader.readLine()) != null){
                String[] users = line.trim().split("\t");
                String uid = users[0];
                count++;
                if(!m_userIDIndex.containsKey(users[0])){
                    System.err.println("The user does not exist in user set!");
                    continue;
                }
                String[] interactions = Arrays.copyOfRange(users, 1, users.length);
                if(interactions.length == 0) continue;
                for(String in: interactions){
                    if(m_userIDIndex.containsKey(in)){
                        if(!m_networkMap.containsKey(uid))
                            m_networkMap.put(uid, new HashSet<>());
                        if(!m_networkMap.containsKey(in))
                            m_networkMap.put(in, new HashSet<>());
                        m_networkMap.get(uid).add(in);
                        m_networkMap.get(in).add(uid);
                    }
                }
            }
            // set the friends for each user
            for(String ui: m_networkMap.keySet()){
                avgFriendSize += m_networkMap.get(ui).size();
                String[] frds = hashSet2Array(m_networkMap.get(ui));
                m_users.get(m_userIDIndex.get(ui)).setFriends(frds);
            }
            reader.close();
            avgFriendSize /= count;
            System.out.format("[Info]Total user size: %d, users with friends: %d, avg friend: %.3f.\n", count,
                    m_networkMap.size(), avgFriendSize);
        } catch(IOException e){
            e.printStackTrace();
        }
    }

    protected String[] hashSet2Array(HashSet<String> strs){
        String[] arr = new String[strs.size()];
        int index = 0;
        for(String str: strs){
            arr[index++] = str;
        }
        return arr;
    }

    // shuffle the whole corpus and save the index information for later use
    public void saveCVIndex(int k, String filename){
        m_corpus.setMasks();
        m_corpus.shuffle(k);
        int[] masks = m_corpus.getMasks();
        ArrayList<_Doc> docs = m_corpus.getCollection();
        try{
            PrintWriter writer = new PrintWriter(new File(filename));
            for(int i=0; i<docs.size(); i++){
                _Review r = (_Review) docs.get(i);
                writer.write(String.format("%s,%d,%d\n", r.getUserID(), r.getID(), masks[i]));
            }
            writer.close();
        } catch(IOException e){
            e.printStackTrace();
        }
    }

    @Override
    public void constructUserIDIndex(){
        m_userIDIndex = new HashMap<String, Integer>();
        for(int i=0; i<m_users.size(); i++){
            m_userIDIndex.put(m_users.get(i).getUserID(), i);
        }
    }

    // load cv index for all the documents
    public void loadCVIndex(String filename){
        try {
            File file = new File(filename);
            BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(file), "UTF-8"));
            String line;
            while((line = reader.readLine()) != null) {
                String[] strs = line.trim().split(",");
                String uId = strs[0];
                int id = Integer.valueOf(strs[1]);
                int mask = Integer.valueOf(strs[2]);

                if(!m_userIDIndex.containsKey(uId))
                    System.out.println("No such user!");
                else {
                    int uIndex = m_userIDIndex.get(uId);
                    if (uIndex > m_users.size())
                        System.out.println("Exceeds the array size!");
                    else {
                        m_users.get(m_userIDIndex.get(uId)).getReviewByID(id).setMask4CV(mask);
                    }
                }
            }
        } catch(IOException e){
            e.printStackTrace();
        }
    }

    public void printDocs4Plane(String filename){
        try{
            PrintWriter writer = new PrintWriter(new File(filename));
            for(_User user: m_users){
                ArrayList<_SparseFeature[]> vectors = new ArrayList<>();
                for(_Review r: user.getReviews()){
                    vectors.add(r.getSparse());
                }
                _SparseFeature[] fvs = Utils.mergeSpVcts(vectors);
                for(_SparseFeature fv: fvs){
                    int index = fv.getIndex();
                    double val = fv.getValue();
                    for(int i=0; i<val; i++){
                        writer.write(index+" ");
                    }
                }
                writer.write("\n");
            }
            writer.close();
            System.out.println("Finish writing docs for PLANE!!");
        } catch(IOException e){
            e.printStackTrace();
        }
    }

//    // print out the network as an adjacency matrix with index in the friend file.
//    public void printNetwork4Plane(String filename){
//        try{
//            PrintWriter writer = new PrintWriter(new File(filename));
//
//            for(int i=0; i<m_users.size(); i++) {
//
//                String uid = m_userIds.get(i);
//                _User user = m_users.get(m_userIDIndex.get(uid));
//                if (user.getFriends() != null && user.getFriends().length > 0) {
//                    for (String frd : user.getFriends()) {
//                        writer.write(m_userMap4Network.get(frd) + "\t");
//                    }
//                    writer.write("\n");
//                }
//            }
//            writer.close();
//
//        } catch(IOException e){
//            e.printStackTrace();
//        }
//    }
//
//    // print out the network as an adjacency matrix with index in the friend file.
//    public void printAdjacencyMatrix(String filename){
//        try{
//            PrintWriter writer = new PrintWriter(new File(filename));
//            for(int i=0; i<m_userIds.size(); i++) {
//                String uid = m_userIds.get(i);
//                _User user = m_users.get(m_userIDIndex.get(uid));
//                if (user.getFriends() != null && user.getFriends().length > 0) {
//                    writer.write(i + "\t");
//                    for (String frd : user.getFriends()) {
//                        writer.write(m_userMap4Network.get(frd) + "\t");
//                    }
//                    writer.write("\n");
//                }
//            }
//            writer.close();
//            System.out.println("Finish saving adj matrix for ");
//        } catch(IOException e){
//            e.printStackTrace();
//        }
//    }
}

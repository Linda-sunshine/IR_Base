package Analyzer;

import opennlp.tools.util.InvalidFormatException;
import structures._Doc;
import structures._Review;
import structures._SparseFeature;
import structures._User;
import utils.Utils;

import java.io.*;
import java.util.*;

/**
 * @author Lin Gong (lg5bt@virginia.edu)
 * The analyzer aims at the directional network information.
 */
public class MultiThreadedNetworkAnalyzer extends MultiThreadedLinkPredAnalyzer {

    Random m_rand = new Random();
    HashMap<String, HashSet<String>> m_networkMap = new HashMap<String, HashSet<String>>();

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

            double totalEdges = 0;

            // load the interactions first
            while((line = reader.readLine()) != null){
                String[] users = line.trim().split("\t");
                String uid = users[0];
                if(!m_userIDIndex.containsKey(users[0])){
                    System.err.println("The user does not exist in user set!");
                    continue;
                }
                String[] interactions = Arrays.copyOfRange(users, 1, users.length);
                if(interactions.length == 0) continue;
                for(String in: interactions){
                    if(in.equals(uid)) continue;
                    if(m_userIDIndex.containsKey(in)){
                        if(!m_networkMap.containsKey(uid))
                            m_networkMap.put(uid, new HashSet<String>());
                        if(!m_networkMap.containsKey(in))
                            m_networkMap.put(in, new HashSet<String>());
                        m_networkMap.get(uid).add(in);
                        m_networkMap.get(in).add(uid);
                    }
                }
            }
            int missing = 0;
            for(String ui: m_networkMap.keySet()){
                for(String frd: m_networkMap.get(ui)){
                    if(!m_networkMap.containsKey(frd))
                        missing++;
                }
            }

            System.out.println("Some edges are not in the set: " + missing);
            // set the friends for each user
            for(String ui: m_networkMap.keySet()){
                totalEdges += m_networkMap.get(ui).size();
                String[] frds = hashSet2Array(m_networkMap.get(ui));
                m_users.get(m_userIDIndex.get(ui)).setFriends(frds);
            }
            reader.close();
            System.out.format("[Info]Total user size: %d, total doc size: %d, users with friends: %d, total edges: " +
                    "%.3f.\n", m_users.size(), m_corpus.getCollection().size(), m_networkMap.size(), totalEdges);
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

    // shuffle the document index based on each user
    public void saveCVIndex(int k, String filename){
        m_corpus.setMasks();

        for(_User u: m_users){
            setMasks4Reviews(u.getReviews(), k);
        }

        ArrayList<_Doc> docs = m_corpus.getCollection();
        int[] stat = new int[5];
        try{
            PrintWriter writer = new PrintWriter(new File(filename));
            for(int i=0; i<docs.size(); i++){
                _Review r = (_Review) docs.get(i);
                writer.write(String.format("%s,%d,%d\n", r.getUserID(), r.getID(), r.getMask4CV()));
                stat[r.getMask4CV()]++;
            }
            writer.close();
            System.out.println("[Info]Finish writing cv index! Stat as follow:");
            for(int s: stat)
                System.out.print(s + "\t");
        } catch(IOException e){
            e.printStackTrace();
        }
    }

    // set masks for one users' all reviews for CV
    public void setMasks4Reviews(ArrayList<_Review> reviews, int k){
        int[] masks = new int[reviews.size()];
        int res = masks.length / k;
        int threshold = res * k;
        for(int i=0; i<masks.length; i++){
            if(i < threshold){
                masks[i] = i % k;
            } else{
                masks[i] = m_rand.nextInt(k);
            }
        }
        shuffle(masks);
        for(int i=0; i< reviews.size(); i++){
            reviews.get(i).setMask4CV(masks[i]);
        }
    }

    // Fisher-Yates shuffle
    public void shuffle(int[] masks){
        int index, tmp;
        for(int i=masks.length-1; i>=0; i--){
            index = m_rand.nextInt(i+1);
            if(index != 1){
                tmp = masks[index];
                masks[index] = masks[i];
                masks[i] = tmp;
            }
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
    public void loadCVIndex(String filename, int kFold){
        try {
            File file = new File(filename);
            int[] stat = new int[kFold];
            BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(file), "UTF-8"));
            String line;
            while((line = reader.readLine()) != null) {
                String[] strs = line.trim().split(",");
                String uId = strs[0];
                int id = Integer.valueOf(strs[1]);
                int mask = Integer.valueOf(strs[2]);
                stat[mask]++;
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
            System.out.println("[Stat]Stat as follow:");
            for(int s: stat)
                System.out.print(s + "\t");
            System.out.println();
        } catch(IOException e){
            e.printStackTrace();
        }
    }

    /*** The data structure is used for cv index ***/
    class _Edge4CV{

        protected String m_userId = "";
        protected int m_cvIdx = -1; // -1: not assigned; 0-x: fold index

        public _Edge4CV(String uid, int cvIdx){
            m_userId = uid;
            m_cvIdx = cvIdx;
        }
        public String getUserId(){
            return m_userId;
        }
        public int getCVIndex(){
            return m_cvIdx;
        }

    }

    HashMap<String, ArrayList<_Edge4CV>> m_uidInteractionsMap = new HashMap<String, ArrayList<_Edge4CV>>();
    HashMap<String, ArrayList<_Edge4CV>> m_uidNonInteractionsMap = new HashMap<String, ArrayList<_Edge4CV>>();

    // Assign interactions to different folds for CV, try to balance different folds.
    public void assignCVIndex4Network(int kFold, int time){
        System.out.println("[Info]Start CV Index assignment for network....");

        ArrayList<Integer> interactions = new ArrayList<Integer>();
        ArrayList<Integer> nonInteractions = new ArrayList<Integer>();

        int orgTotal = 0, realTotal = 0;
        for(int i=0; i<m_users.size(); i++){
            _User ui = m_users.get(i);
            String uiId = ui.getUserID();
            String[] friends = ui.getFriends();

            interactions.clear();
            nonInteractions.clear();

            // ignore the users without any interactions
            if(friends != null && friends.length > 0) {
                if(!m_uidInteractionsMap.containsKey(uiId))
                    m_uidInteractionsMap.put(uiId, new ArrayList<_Edge4CV>());
                if(!m_uidNonInteractionsMap.containsKey(uiId))
                    m_uidNonInteractionsMap.put(uiId, new ArrayList<_Edge4CV>());

                orgTotal += friends.length;
                // construct the friend indexes
                for(String frd: friends){
                    int frdIdx = m_userIDIndex.get(frd);
                    if(frdIdx > i)
                        interactions.add(frdIdx);
                }

                for(int j=i+1; j<m_users.size(); j++){
                    if(!interactions.contains(j))
                        nonInteractions.add(j);
                }
                // sample masks for interactions: assign fold number to interactiosn
                int[] masks4Interactions = generateMasks(interactions.size(), kFold);
                // collect the interactions in the hashmap
                for(int m=0; m<interactions.size(); m++){
                    String ujId = m_users.get(interactions.get(m)).getUserID();

                    if(!m_uidInteractionsMap.containsKey(ujId))
                        m_uidInteractionsMap.put(ujId, new ArrayList<_Edge4CV>());
                    m_uidInteractionsMap.get(uiId).add(new _Edge4CV(ujId, masks4Interactions[m]));
                    m_uidInteractionsMap.get(ujId).add(new _Edge4CV(uiId, masks4Interactions[m]));
                }

                // sample non-interactions: select non-interactions for each fold, might be repetitive
                HashMap<Integer, HashSet<Integer>> foldNonInteractions = new HashMap<Integer, HashSet<Integer>>();
                for(int k=0; k<kFold; k++){
                    int number = time * interactions.size() / 5;
                    foldNonInteractions.put(k, sampleNonInteractions(nonInteractions, number));
                }
                // collect the non-interactions in the hashmap
                for(int k: foldNonInteractions.keySet()){
                    for(int ujIdx: foldNonInteractions.get(k)){
                        String ujId = m_users.get(ujIdx).getUserID();
                        if(!m_uidNonInteractionsMap.containsKey(ujId))
                            m_uidNonInteractionsMap.put(ujId, new ArrayList<_Edge4CV>());
                        m_uidNonInteractionsMap.get(uiId).add(new _Edge4CV(ujId, k));
                        m_uidNonInteractionsMap.get(ujId).add(new _Edge4CV(uiId, k));
                    }
                }
            }
        }
        System.out.println("Interaction user size: " + m_uidInteractionsMap.size());
        System.out.println("Non-interaction user size: " + m_uidNonInteractionsMap.size());

        for(String uid: m_uidInteractionsMap.keySet()){
            realTotal += m_uidInteractionsMap.get(uid).size();
        }
        System.out.format("Org Total: %d, real Total: %d\n", orgTotal, realTotal);
    }

    public void sanityCheck4CVIndex4Network(boolean interactionFlag){
        HashMap<String, ArrayList<_Edge4CV>> map = interactionFlag ? m_uidInteractionsMap: m_uidNonInteractionsMap;
        if(interactionFlag)
            System.out.println("=====Stat for users' interactions======");
        else
            System.out.println("=====Stat for users' non-interactions======");

        double total = 0, avg = 0;
        double[] stat = new double[5];

        // we only care about users who have interactions, thus use their idx for indexing
        int count = 0;
        for(String uid: m_uidInteractionsMap.keySet()){
            if(map.get(uid).size() == 0) {
                count++;
                continue;
            }

            for(_Edge4CV eg: map.get(uid)){
                total++;
                stat[eg.getCVIndex()]++;
            }
        }
        System.out.format("%d users don't have non-interactions!\n", count);
        avg = total / m_uidInteractionsMap.size();
        System.out.format("[Stat]Total user size: %d, total edge: %.1f, avg interaction/non-interaction size: %.3f\nEach fold's " +
                "interaction/non-interaction size is as follows:\n", m_uidInteractionsMap.size(), total, avg);
        for(double s: stat){
            System.out.print(s+"\t");
        }
        System.out.println("\n");
    }

    public void saveCVIndex4Network(String filename, boolean interactionFlag){
        HashMap<String, ArrayList<_Edge4CV>> map = interactionFlag ? m_uidInteractionsMap : m_uidNonInteractionsMap;
        try{
            // we only care about users who have interactions
            PrintWriter writer = new PrintWriter(new File(filename));
            for(String uid: m_uidInteractionsMap.keySet()){
                for(_Edge4CV eg: map.get(uid)){
                    writer.format("%s,%s,%d\n", uid, eg.getUserId(), eg.getCVIndex());
                }
            }
            writer.close();
            System.out.format("[Info]Finish writing %d users in %s.\n", m_uidInteractionsMap.size(), filename);
        } catch(IOException e){
            e.printStackTrace();
        }
    }


    // set masks for one users' all reviews for CV
    public int[] generateMasks(int len, int k){
        int[] masks = new int[len];

        int res = masks.length / k;
        int threshold = res * k;
        for(int i=0; i<masks.length; i++){
            if(i < threshold){
                masks[i] = i % k;
            } else{
                masks[i] = m_rand.nextInt(k);
            }
        }
        shuffle(masks);
        return masks;
    }

    public HashSet<Integer> sampleNonInteractions(ArrayList<Integer> nonInteractions, int nu){
        HashSet<Integer> sampledNonInteractions = new HashSet<Integer>();
        for(int i=0; i<nu; i++){
            int idx = m_rand.nextInt(nonInteractions.size());
            sampledNonInteractions.add(nonInteractions.get(idx));
        }
        return sampledNonInteractions;
    }

    public void findUserWithMaxDocSize(){
        int max = -1;
        int count = 0;
        for(_User u: m_users){
            if(u.getReviewSize() > 1000) {
                System.out.println(u.getUserID());
                count++;
            }
            max = Math.max(max, u.getReviewSize());
        }
        System.out.println("Max doc size: " + max);
    }

    /****We need to output some files for running baselines
     * TADW, PLANE etc.
     * ****/
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

    public void printData4TADW(String filename){
        try{
            PrintWriter writer = new PrintWriter(new File(filename));
            for(_User user: m_users) {
                user.constructLRSparseVector();
                user.normalizeProfile();
                String uid = user.getUserID();
                for (_SparseFeature fv : user.getBoWProfile()) {
                    writer.write(String.format("%s\t%d\t%.3f\n", uid, fv.getIndex(), fv.getValue()));
                }
            }
            writer.close();
            System.out.format("Finish writing %d users' data for TADW.\n", m_users.size());
        } catch(IOException e){
            e.printStackTrace();
        }
    }

}

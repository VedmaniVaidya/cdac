package Q2;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;

class ReadFile implements Runnable {
    private String sourceFile;
    private FileWriter writer;
    private Object lock;

    public ReadFile(String sourceFile, FileWriter writer, Object lock) {
        this.sourceFile = sourceFile;
        this.writer = writer;
        this.lock = lock;
    }

    public void run() {
        try {
            FileReader reader = new FileReader(sourceFile);
            int character;
            while ((character = reader.read()) != -1) {
                writer.write(character);
            }
            reader.close();
            synchronized (lock) {
                lock.notify();
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}

class WriteFile implements Runnable {
    private String destinationFile;
    private FileWriter writer;
    private Object lock;

    public WriteFile(String destinationFile, FileWriter writer, Object lock) {
        this.destinationFile = destinationFile;
        this.writer = writer;
        this.lock = lock;
    }

    public void run() {
        try {
            synchronized (lock) {
                lock.wait();
            }
            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }
}

public class Main {
    public static void main(String[] args) {
        String sourceFile = "/home/dai/java/assignment_11/src/Q2/src.txt";
        String destinationFile = "/home/dai/java/assignment_11/src/Q2/dest.txt";
        Object lock = new Object();

        try {
            FileWriter writer = new FileWriter(destinationFile, true);
            Thread readThread = new Thread(new ReadFile(sourceFile, writer, lock));
            Thread writeThread = new Thread(new WriteFile(destinationFile, writer, lock));
            readThread.start();
            writeThread.start();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}

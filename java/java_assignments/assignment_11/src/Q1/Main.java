package Q1;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Scanner;

public class Main {
    public static void main(String[] args) {
        try {
            // Create a Scanner object to read the file
            File inputFile = new File("/home/dai/java/assignment_11/src/Q1/src.txt");
            Scanner scanner = new Scanner(inputFile);

            // Create a FileWriter object to write to the new file
            File outputFile = new File("/home/dai/java/assignment_11/src/Q1/dest.txt");
            FileWriter writer = new FileWriter(outputFile);

            // Read each line from the input file and write it to the output file
            while (scanner.hasNextLine()) {
                String line = scanner.nextLine();
                writer.write(line);
                writer.write(System.lineSeparator()); // Add line separator after each line
            }

            // Close the Scanner and FileWriter objects
            scanner.close();
            writer.close();

            System.out.println("New file created successfully.");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}

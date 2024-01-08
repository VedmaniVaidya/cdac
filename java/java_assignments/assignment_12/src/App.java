import java.io.*;
import java.util.ArrayList;
import java.util.Scanner;

public class App {
    private static final String FILE_NAME = "src/account_data.dat";
    private static ArrayList<Account> accounts = new ArrayList<>();

    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        int choice;

        do {
            System.out.println("Menu:");
            System.out.println("1. Add an account");
            System.out.println("2. Display all accounts");
            System.out.println("3. Deposit amount");
            System.out.println("4. Withdraw amount");
            System.out.println("5. Display accounts with balance less than 20000");
            System.out.println("6. Exit");
            System.out.print("Enter your choice: ");
            choice = scanner.nextInt();

            switch (choice) {
                case 1:
                    addAccount();
                    break;
                case 2:
                    displayAllAccounts();
                    break;
                case 3:
                    depositAmount();
                    break;
                case 4:
                    withdrawAmount();
                    break;
                case 5:
                    displayAccountsLessThan20000();
                    break;
                case 6:
                    System.out.println("Exiting...");
                    break;
                default:
                    System.out.println("Invalid choice. Please try again.");
            }
        } while (choice != 6);
    }

    private static void addAccount() {
        Scanner scanner = new Scanner(System.in);

        System.out.print("Enter account number: ");
        int accountNumber = scanner.nextInt();
        scanner.nextLine(); // Consume newline character

        System.out.print("Enter name: ");
        String name = scanner.nextLine();

        System.out.print("Enter balance: ");
        double balance = scanner.nextDouble();

        Account account = new Account(accountNumber, name, balance);
        accounts.add(account);

        try (ObjectOutputStream outputStream = new ObjectOutputStream(new FileOutputStream(FILE_NAME))) {
            outputStream.writeObject(accounts);
            System.out.println("Account added successfully.");
        } catch (IOException e) {
            System.out.println("Error occurred while saving account data.");
        }
    }

    private static void displayAllAccounts() {
        try (ObjectInputStream inputStream = new ObjectInputStream(new FileInputStream(FILE_NAME))) {
            accounts = (ArrayList<Account>) inputStream.readObject();

            accounts.forEach(account -> System.out.println(account));
        } catch (IOException | ClassNotFoundException e) {
            System.out.println("Error occurred while reading account data.");
        }
    }

    private static void depositAmount() {
        Scanner scanner = new Scanner(System.in);

        System.out.print("Enter account number: ");
        int accountNumber = scanner.nextInt();

        Account account = findAccount(accountNumber);
        if (account != null) {
            System.out.print("Enter amount to deposit: ");
            double amount = scanner.nextDouble();

            double newBalance = account.getBalance() + amount;
            account.setBalance(newBalance);

            updateAccountData();
            System.out.println("Amount deposited successfully.");
        } else {
            System.out.println("Account not found.");
        }
    }

    private static void withdrawAmount() {
        Scanner scanner = new Scanner(System.in);

        System.out.print("Enter account number: ");
        int accountNumber = scanner.nextInt();

        Account account = findAccount(accountNumber);
        if (account != null) {
            System.out.print("Enter amount to withdraw: ");
            double amount = scanner.nextDouble();

            if (amount <= account.getBalance()) {
                double newBalance = account.getBalance() - amount;
                account.setBalance(newBalance);

                updateAccountData();
                System.out.println("Amount withdrawn successfully.");
            } else {
                System.out.println("Insufficient balance.");
            }
        } else {
            System.out.println("Account not found.");
        }
    }

    private static void displayAccountsLessThan20000() {
        try (ObjectInputStream inputStream = new ObjectInputStream(new FileInputStream(FILE_NAME))) {
            accounts = (ArrayList<Account>) inputStream.readObject();

            accounts.stream()
                    .filter(account -> account.getBalance() < 20000)
                    .forEach(account -> System.out.println(account));
        } catch (IOException | ClassNotFoundException e) {
            System.out.println("Error occurred while reading account data.");
        }
    }

    private static Account findAccount(int accountNumber) {
        for (Account account : accounts) {
            if (account.getAccountNumber() == accountNumber) {
                return account;
            }
        }
        return null;
    }

    private static void updateAccountData() {
        try (ObjectOutputStream outputStream = new ObjectOutputStream(new FileOutputStream(FILE_NAME))) {
            outputStream.writeObject(accounts);
        } catch (IOException e) {
            System.out.println("Error occurred while updating account data.");
        }
    }
}

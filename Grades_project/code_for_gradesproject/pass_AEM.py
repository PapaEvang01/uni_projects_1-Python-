"""
===========================================
 Student Grades Analyzer - Python Project
===========================================

This program reads a list of student IDs and their grades from a text file,
analyzes the data, and provides useful insights such as:

- Total number of students
- Number and percentage of students who passed (grade >= 5)
- Average grade (overall and for those who passed)
- Minimum and maximum grade
- Visualizations (histogram & pass/fail bar chart)
- Saving passed students to a separate file
- Student lookup: allows a student to check their grade and pass/fail status

Designed to work with grade files using either dot (.) or comma (,) as decimal separators.

Language: Python 3.12
Libraries: matplotlib

"""

# Importing matplotlib for plotting
import matplotlib.pyplot as plt

# Function to load grades from the provided text file
def load_grades_from_file(filename):
    grades = []          # List of just the numeric grades
    student_data = []    # List of (ID, grade) pairs
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) == 2:
                try:
                    grade = float(parts[1].replace(",", "."))  # Convert "5,5" to 5.5
                    grades.append(grade)
                    student_data.append((parts[0], grade))     # Store (student_id, grade)
                except ValueError:
                    continue  # Skip lines with invalid data
    return grades, student_data

# Function to analyze statistics from the grade list
def analyze_grades(grades):
    total = len(grades)
    passed_grades = [g for g in grades if g >= 5]
    passed = len(passed_grades)

    # Calculate averages and pass rate
    overall_avg = sum(grades) / total if total else 0
    passed_avg = sum(passed_grades) / passed if passed else 0
    pass_percent = (passed / total) * 100 if total else 0

    return {
        "total": total,
        "passed": passed,
        "pass_percent": pass_percent,
        "average": overall_avg,
        "passed_average": passed_avg,
        "min": min(grades) if grades else 0,
        "max": max(grades) if grades else 0
    }

# Function that allows the user to search for their grade by student number
def search_student(student_data):
    print("\n--- Student Grade Lookup ---")
    query = input("Enter your student number (e.g., 58324): ").strip()

    found = False
    for student_id, grade in student_data:
        if student_id == query:
            found = True
            status = "PASSED ✅" if grade >= 5 else "FAILED ❌"
            print(f"Student ID: {student_id} | Grade: {grade:.2f} | Status: {status}")
            break

    if not found:
        print("Student number not found.")

# Function to save all passed students to a text file
def save_passed_students(student_data, filename="passed_students.txt"):
    with open(filename, 'w', encoding='utf-8') as f:
        for student_id, grade in student_data:
            if grade >= 5:
                f.write(f"{student_id} {grade:.2f}\n")

# Function to plot a histogram of all grades
def plot_grade_histogram(grades):
    plt.figure()
    plt.hist(grades, bins=20, edgecolor='black', color='skyblue')
    plt.title("Grade Distribution")
    plt.xlabel("Grade")
    plt.ylabel("Number of Students")
    plt.grid(True)
    plt.show()

# Function to plot a simple bar chart: passed vs failed
def plot_pass_fail_bar(total, passed):
    failed = total - passed
    plt.figure()
    plt.bar(["Passed", "Failed"], [passed, failed], color=["green", "red"])
    plt.title("Pass vs Fail")
    plt.ylabel("Number of Students")
    plt.show()

# Main driver function
def main():
    # Load data from the .txt file
    filename = r"C:\Users\VaggelisPaps\PycharmProjects\who_pass_the_exam\GRADES.TXT"
    grades, student_data = load_grades_from_file(filename)
    stats = analyze_grades(grades)

    # Print statistics
    print(f"Total students: {stats['total']}")
    print(f"Passed students: {stats['passed']}")
    print(f"Pass percentage: {stats['pass_percent']:.2f}%")
    print(f"Average grade (overall): {stats['average']:.2f}")
    print(f"Average grade (passed only): {stats['passed_average']:.2f}")
    print(f"Minimum grade: {stats['min']:.2f}")
    print(f"Maximum grade: {stats['max']:.2f}")

    # Allow the user to search for their own grade
    search_student(student_data)

    # Save passed students to file
    save_passed_students(student_data)

    # Plot graphs
    plot_grade_histogram(grades)
    plot_pass_fail_bar(stats['total'], stats['passed'])

# Start the program
if __name__ == "__main__":
    main()

"""
====================================================
 Random Grade Generator + Analyzer for Student Data
====================================================

This program:
1. Generates random fake grades for students (IDs 5XXXXX)
2. Saves the grades to FAKE_GRADES.TXT
3. Loads and analyzes the data
4. Displays statistics and charts
5. Allows user to search for their grade by ID

Author: Vaggelis Paps
"""

import random
import matplotlib.pyplot as plt

# --------------------------------------
# GENERATE RANDOM STUDENT GRADES
# --------------------------------------
def generate_random_grades(filename, num_students=200):
    with open(filename, 'w', encoding='utf-8') as f:
        for _ in range(num_students):
            student_id = str(random.randint(50000, 59999))
            grade = random.randint(0, 20) / 2.0  # grades from 0.0 to 10.0 in 0.5 steps
            grade_str = str(grade).replace('.', ',')
            f.write(f"{student_id} {grade_str}\n")
    print(f"{num_students} random student grades saved to '{filename}'.")

# --------------------------------------
# LOAD GRADES FROM FILE
# --------------------------------------
def load_grades_from_file(filename):
    grades = []
    student_data = []
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) == 2:
                try:
                    grade = float(parts[1].replace(",", "."))
                    grades.append(grade)
                    student_data.append((parts[0], grade))
                except ValueError:
                    continue
    return grades, student_data

# --------------------------------------
# ANALYZE GRADE DATA
# --------------------------------------
def analyze_grades(grades):
    total = len(grades)
    passed_grades = [g for g in grades if g >= 5]
    passed = len(passed_grades)
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

# --------------------------------------
# SEARCH FOR STUDENT ID
# --------------------------------------
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

# --------------------------------------
# SAVE PASSED STUDENTS
# --------------------------------------
def save_passed_students(student_data, filename="passed_students.txt"):
    with open(filename, 'w', encoding='utf-8') as f:
        for student_id, grade in student_data:
            if grade >= 5:
                f.write(f"{student_id} {grade:.2f}\n")

# --------------------------------------
# PLOTTING FUNCTIONS
# --------------------------------------
def plot_grade_histogram(grades):
    plt.figure()
    plt.hist(grades, bins=20, edgecolor='black', color='skyblue')
    plt.title("Grade Distribution")
    plt.xlabel("Grade")
    plt.ylabel("Number of Students")
    plt.grid(True)
    plt.show()

def plot_pass_fail_bar(total, passed):
    failed = total - passed
    plt.figure()
    plt.bar(["Passed", "Failed"], [passed, failed], color=["green", "red"])
    plt.title("Pass vs Fail")
    plt.ylabel("Number of Students")
    plt.show()

# --------------------------------------
# MAIN FUNCTION
# --------------------------------------
def main():
    filename = r"C:\Users\VaggelisPaps\PycharmProjects\who_pass_the_exam\FAKE_GRADES.TXT"

    # Generate and save fake data
    generate_random_grades(filename)

    # Load and analyze data
    grades, student_data = load_grades_from_file(filename)
    stats = analyze_grades(grades)

    # Print stats
    print(f"Total students: {stats['total']}")
    print(f"Passed students: {stats['passed']}")
    print(f"Pass percentage: {stats['pass_percent']:.2f}%")
    print(f"Average grade (overall): {stats['average']:.2f}")
    print(f"Average grade (passed only): {stats['passed_average']:.2f}")
    print(f"Minimum grade: {stats['min']:.2f}")
    print(f"Maximum grade: {stats['max']:.2f}")

    # Search student and visualize
    search_student(student_data)
    save_passed_students(student_data)
    plot_grade_histogram(grades)
    plot_pass_fail_bar(stats['total'], stats['passed'])

# Run everything
if __name__ == "__main__":
    main()

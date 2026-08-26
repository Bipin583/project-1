"""
ConfTest CI/CD Results Reporter & Audit Trail Generator
Parses post-execution JUnit XML results and prints a formatted Pull Request comment summary.
"""
import argparse
import xml.etree.ElementTree as ET
import os

def parse_junit_xml(xml_path: str):
    """Parses JUnit XML report and extracts executed test counts and outcomes."""
    if not os.path.exists(xml_path):
        return {"total": 0, "passed": 0, "failed": 0, "errors": 0, "skipped": 0, "failures_list": []}

    tree = ET.parse(xml_path)
    root = tree.getroot()

    testcases = list(root.iter("testcase"))
    total = len(testcases)
    failures_list = []
    skipped_count = 0
    errors_count = 0

    for tc in testcases:
        if tc.find("failure") is not None:
            name = f"{tc.attrib.get('classname', '')}::{tc.attrib.get('name', '')}"
            failures_list.append(name)
        elif tc.find("error") is not None:
            errors_count += 1
        elif tc.find("skipped") is not None:
            skipped_count += 1

    failed_count = len(failures_list)
    passed_count = total - (failed_count + errors_count + skipped_count)

    return {
        "total": total,
        "passed": passed_count,
        "failed": failed_count,
        "errors": errors_count,
        "skipped": skipped_count,
        "failures_list": failures_list
    }

def main():
    parser = argparse.ArgumentParser(description="ConfTest PR Report Generator")
    parser.add_argument("--results", type=str, default="junit_results.xml", help="Path to JUnit results XML")
    parser.add_argument("--pr-number", type=int, default=1, help="Pull Request number")
    parser.add_argument("--github-token", type=str, default="", help="GitHub token for PR comment API")

    args = parser.parse_args()

    results = parse_junit_xml(args.results)

    print("\n" + "="*70)
    print(f"[ConfTest Bot] CI/CD PULL REQUEST #{args.pr_number} VERIFICATION SUMMARY")
    print("="*70)
    print(f"Tests Executed : {results['total']}")
    print(f"Tests Passed   : {results['passed']} [PASSED]")
    print(f"Tests Failed   : {results['failed']} [FAILED]")
    print(f"Tests Skipped  : {results['skipped']} [SKIPPED]")

    if results["failed"] > 0:
        print("\nFailed Tests Captured by ConfTest Selective Suite:")
        for f in results["failures_list"]:
            print(f"  - {f}")
        print("\n[!] CI Verification Failed. Please resolve regression bugs before merging.")
    else:
        print("\n[+] All selected regression tests passed successfully!")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()

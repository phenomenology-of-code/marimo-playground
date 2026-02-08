import marimo

__generated_with = "0.19.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This projects loads marimo using the UV projects method https://docs.marimo.io/guides/package_management/using_uv/#using-uv-projects
    """)
    return


@app.cell
def _():
    def fizzbuzz(n):
        result = []
        for i in range(1, n + 1):
            if i % 3 == 0 and i % 5 == 0:
                result.append("FizzBuzz")
            elif i % 3 == 0:
                result.append("Fizz")
            elif i % 5 == 0:
                result.append("Buzz")
            else:
                result.append(str(i))
        return result

    # Generate FizzBuzz results for first 100 numbers
    fizzbuzz_results = fizzbuzz(100)
    return (fizzbuzz_results,)


@app.cell
def _(fizzbuzz_results):
    import matplotlib.pyplot as plt

    # Create a bar chart to visualize FizzBuzz results
    categories = ['Fizz', 'Buzz', 'FizzBuzz', 'Numbers']
    counts = [fizzbuzz_results.count('Fizz'), fizzbuzz_results.count('Buzz'), fizzbuzz_results.count('FizzBuzz'), len(fizzbuzz_results) - fizzbuzz_results.count('Fizz') - fizzbuzz_results.count('Buzz') - fizzbuzz_results.count('FizzBuzz')]

    plt.figure(figsize=(10, 6))
    plt.bar(categories, counts, color=['#FF6347', '#4682B4', '#228B22', '#8B4513'])
    plt.title('FizzBuzz Distribution (First 100 Numbers)')
    plt.xlabel('Category')
    plt.ylabel('Count')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.gca()
    return


if __name__ == "__main__":
    app.run()

document.addEventListener("DOMContentLoaded", async () => {
  const outputDiv = document.getElementById("output");
  const commentsListDiv = document.getElementById("comments-list");
  const pieContainer = document.getElementById("piechart-container");
  const pieChartCanvas = document.getElementById("piechart");
  const API_KEY = "AIzaSyAgmJrE8oqan69HPqizGgjoxWL_oERDA-k"; // <-- Replace with your YouTube Data API key
  const SENTIMENT_API_URL = "http://localhost:5000/predict";
  let pieChart = null;

  if (!outputDiv || !commentsListDiv || !pieContainer || !pieChartCanvas) {
    alert("UI elements not found. Please check popup.html structure.");
    return;
  }

  chrome.tabs.query({ active: true, currentWindow: true }, async (tabs) => {
    const url = tabs[0].url;
    const youtubeRegex = /^https:\/\/(?:www\.)?youtube\.com\/watch\?v=([\w-]{11})/;
    const match = url.match(youtubeRegex);

    if (match && match[1]) {
      const videoId = match[1];
      outputDiv.textContent = `YouTube Video ID: ${videoId}\nFetching comments...`;

      const comments = await fetchComments(videoId);
      if (!comments || comments.length === 0) {
        outputDiv.textContent += "\nNo comments found for this video.";
        pieContainer.style.display = "none";
        commentsListDiv.innerHTML = "";
        return;
      }

      outputDiv.textContent += `\nFound ${comments.length} comments. Analyzing...`;

      const predictions = await getSentimentPredictions(comments);

      if (predictions && predictions.length === comments.length) {
        // Count sentiments
        const sentimentCounts = { 1: 0, 0: 0, 2: 0 };
        predictions.forEach((sentiment) => {
          sentimentCounts[sentiment]++;
        });
        const total = predictions.length;
        const positivePercentage = ((sentimentCounts["1"] / total) * 100).toFixed(2);
        const negativePercentage = ((sentimentCounts["0"] / total) * 100).toFixed(2);
        const neutralPercentage = ((sentimentCounts["2"] / total) * 100).toFixed(2);

        outputDiv.textContent = `Sentiment Analysis Results:\nPositive: ${positivePercentage}%\nNegative: ${negativePercentage}%\nNeutral: ${neutralPercentage}%`;

        // Show pie chart
        pieContainer.style.display = "flex";
        if (pieChart) pieChart.destroy();
        pieChart = new Chart(pieChartCanvas, {
          type: "pie",
          data: {
            labels: ["Positive", "Negative", "Neutral"],
            datasets: [{
              data: [sentimentCounts["1"], sentimentCounts["0"], sentimentCounts["2"]],
              backgroundColor: [
                "rgba(67,233,123,0.85)",
                "rgba(255,88,88,0.85)",
                "rgba(189,189,189,0.85)"
              ],
              borderWidth: 2,
              borderColor: "#fff"
            }]
          },
          options: {
            plugins: {
              legend: {
                display: true,
                position: "bottom"
              }
            }
          }
        });

        // Show top 10 comments with colored backgrounds and zoom effect
        commentsListDiv.innerHTML = "";
        const topComments = comments.slice(0, 10);
        topComments.forEach((comment, idx) => {
          const sentiment = predictions[idx];
          let sentimentClass = "sentiment-neutral";
          if (sentiment === 1) sentimentClass = "sentiment-positive";
          else if (sentiment === 0) sentimentClass = "sentiment-negative";
          const div = document.createElement("div");
          div.className = `comment-item ${sentimentClass}`;
          div.textContent = `${idx + 1}. ${comment}`;
          commentsListDiv.appendChild(div);
        });
      } else {
        outputDiv.textContent += "\nError: Prediction failed or mismatched.";
        pieContainer.style.display = "none";
        commentsListDiv.innerHTML = "";
      }
    } else {
      outputDiv.textContent = "This is not a YouTube video page to use this extension.";
      pieContainer.style.display = "none";
      commentsListDiv.innerHTML = "";
    }
  });

  async function fetchComments(videoId) {
    let comments = [];
    let pageToken = "";
    try {
      while (comments.length < 100) {
        // Limit to 100 comments
        const response = await fetch(
          `https://www.googleapis.com/youtube/v3/commentThreads?part=snippet&videoId=${videoId}&maxResults=100&pageToken=${pageToken}&key=${API_KEY}`
        );
        const data = await response.json();
        if (!data.items) break;
        data.items.forEach((item) => {
          comments.push(item.snippet.topLevelComment.snippet.textOriginal);
        });
        pageToken = data.nextPageToken;
        if (!pageToken) break; // No more pages
      }
    } catch (error) {
      console.error("Error fetching comments: ", error);
    }
    return comments;
  }

  async function getSentimentPredictions(comments) {
    try {
      const response = await fetch(SENTIMENT_API_URL, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ comments }),
      });
      const result = await response.json();
      // If your backend returns [{comment, sentiment}], map to sentiment
      if (Array.isArray(result) && typeof result[0] === "object" && "sentiment" in result[0]) {
        return result.map((item) => item.sentiment);
      }
      // If your backend returns [0,1,2,...]
      return result;
    } catch (error) {
      console.error("Error fetching prediction: ", error);
      outputDiv.textContent += "\nError fetching sentiment predictions.";
      return [];
    }
  }
});
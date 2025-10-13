//
//  NetworkManager.swift
//  RanKing
//
//  Created by Jonathan Da Silva on 10/13/25.
//

import Foundation

// An enumeration for handling different network errors.
enum NetworkError: Error {
    case invalidURL
    case invalidResponse
    case decodingError
}

@MainActor // Ensures that updates to @Published properties happen on the main thread.
class NetworkManager: ObservableObject {
    // This property will be published to any subscribed SwiftUI views.
    @Published var posts: [String] = []

    // MARK: - GET Request
    func fetchPosts() async throws {
        guard let url = URL(string: "https://jsonplaceholder.typicode.com/posts") else {
            throw NetworkError.invalidURL
        }

        // 1. Await the data and the URL response from the URLSession data task.
        let (data, response) = try await URLSession.shared.data(from: url)

        // 2. Check for a valid HTTP response.
        guard let httpResponse = response as? HTTPURLResponse, httpResponse.statusCode == 200 else {
            throw NetworkError.invalidResponse
        }

        // 3. Decode the received JSON data into an array of Post objects.
        do {
            self.posts = try JSONDecoder().decode([String].self, from: data)
        } catch {
            throw NetworkError.decodingError
        }
    }

    // MARK: - POST Request
    func createPost(title: String, userId: Int) async throws -> String? {
        guard let url = URL(string: "https://jsonplaceholder.typicode.com/posts") else {
            throw NetworkError.invalidURL
        }

        // 1. Create a URLRequest and configure it for a POST request.
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")

        // 2. Create a dictionary to hold the post data.
        let newPost = ["HI"]

        // 3. Encode the dictionary into JSON data and set it as the request body.
        request.httpBody = try JSONEncoder().encode(newPost)

        // 4. Await the data and response from the session.
        let (data, response) = try await URLSession.shared.data(for: request)

        // 5. Check for a valid HTTP response.
        guard let httpResponse = response as? HTTPURLResponse, httpResponse.statusCode == 201 else {
            throw NetworkError.invalidResponse
        }
        
        // 6. Decode the server's response (often returns the created object with an ID).
        do {
            let createdPost = try JSONDecoder().decode(String.self, from: data)
            return createdPost
        } catch {
            throw NetworkError.decodingError
        }
    }
}

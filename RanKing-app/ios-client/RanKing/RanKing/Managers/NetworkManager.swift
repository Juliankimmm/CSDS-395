//
//  NetworkManager.swift
//  RanKing
//
//  Created by Jonathan Da Silva on 10/13/25.
//

import Foundation
import SwiftUI

// An enumeration for handling different network errors.
enum NetworkError: Error {
    case invalidURL
    case invalidResponse
    case decodingError
}

@MainActor // Ensures that updates to @Published properties happen on the main thread.
class NetworkManager: ObservableObject {
    
    public static var networkManager = NetworkManager()
    
    // MARK: - GET Request
    func fetchContests(query: String? = "") async throws -> [Contest]? {
        //TODO add query if necessary
        guard let url = URL(string: "http://localhost:8585/contests/") else {
            throw NetworkError.invalidURL
        }
        let (data, response) = try await URLSession.shared.data(from: url)
        print(String(data: data, encoding: .utf8) ?? "No data")

        guard let httpResponse = response as? HTTPURLResponse, httpResponse.statusCode == 200 else {
            throw NetworkError.invalidResponse
        }
        
        print("200!")
        var contests: [Contest]? = nil
        do {
            contests = try JSONDecoder().decode([Contest].self, from: data)
        } catch {
            print("Decoding Error :( \(error)")
            throw NetworkError.decodingError
        }
        
        print("DECODED!!! \(contests ?? [])")
        return contests
    }
    
    func fetchSumbissions(contestId: Int) async throws -> [SubmissionResponse]? {
        guard let url = URL(string: "http://localhost:8585/contests/\(contestId)/submissions") else {
            throw NetworkError.invalidURL
        }
        let (data, response) = try await URLSession.shared.data(from: url)
        print(String(data: data, encoding: .utf8) ?? "No data")
        guard let httpResponse = response as? HTTPURLResponse, httpResponse.statusCode == 200 else {
            throw NetworkError.invalidResponse
        }
        
        
        var submissions : [SubmissionResponse]? = nil
        do {
            submissions = try JSONDecoder().decode([SubmissionResponse].self, from: data)
        } catch {
            throw NetworkError.decodingError
        }
        return submissions
    }

    // MARK: - POST Request
    func register(username: String, email: String, password: String) async throws -> User? {
        guard let url = URL(string: "http://localhost/register") else {
            throw NetworkError.invalidURL
        }

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")

        let userData = try JSONSerialization.data(withJSONObject: ["username": username, "email": email, "password": password], options: [])

        request.httpBody = userData

        let (data, response) = try await URLSession.shared.data(for: request)
        print(String(data: data, encoding: .utf8) ?? "No data")

        guard let httpResponse = response as? HTTPURLResponse, httpResponse.statusCode == 201 else {
            throw NetworkError.invalidResponse
        }
        
        
        do {
            let createdPost = try JSONDecoder().decode(User.self, from: data)
            return createdPost
        } catch {
            throw NetworkError.decodingError
        }
    }
    
    func login(email: String, password: String) async throws -> Token? {
        guard let url = URL(string: "http://localhost:8585/login") else {
            throw NetworkError.invalidURL
        }
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        let userData = try JSONSerialization.data(withJSONObject: ["email": email, "password": password], options: [])
        request.httpBody = userData
        let (data, response) = try await URLSession.shared.data(for: request)

        print(String(data: data, encoding: .utf8) ?? "No data")
        guard let httpResponse = response as? HTTPURLResponse, httpResponse.statusCode == 201 else {
            
            throw NetworkError.invalidResponse
        }
        
        
        do {
            let createdPost = try JSONDecoder().decode(Token.self, from: data)
            return createdPost
        } catch {
            print("\(error)")
            throw NetworkError.decodingError
        }
    }
    
    func sendSubmission(imageData: Data, contestId: Int, accessToken: String) async throws -> SubmissionResponse {
        guard let url = URL(string: "http://localhost:8585/contests/\(contestId)/submissions") else {
            throw NetworkError.invalidURL
        }
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("Bearer \(accessToken)", forHTTPHeaderField: "Authorization")
        let boundary = "Boundary-\(UUID().uuidString)"
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
        request.httpBody = createMultipartBody(
            boundary: boundary,
            fieldName: "file",
            fileName: "submission.jpg",
            mimeType: "image/jpeg",
            fileData: imageData
        )
        do {
            print("Got here")
            let (data, response) = try await URLSession.shared.data(for: request)
            print(String(data: data, encoding: .utf8) ?? "No data")
            guard let httpResponse = response as? HTTPURLResponse, httpResponse.statusCode == 201 else {
                throw NetworkError.invalidResponse
            }
            do {
                let createdSubmission = try JSONDecoder().decode(SubmissionResponse.self, from: data)
                return createdSubmission
            } catch {
                print("Error decoding submission: \(error)")
                throw NetworkError.decodingError
            }
        } catch {
            print("Error sending submission: \(error)")
            throw NetworkError.invalidResponse
        }
    }
    
    private func createMultipartBody(
        boundary: String,
        fieldName: String,
        fileName: String,
        mimeType: String,
        fileData: Data
    ) -> Data {
        var body = Data()
        body.append("--\(boundary)\r\n".data(using: .utf8)!)
        body.append("Content-Disposition: form-data; name=\"\(fieldName)\"; filename=\"\(fileName)\"\r\n".data(using: .utf8)!)
        body.append("Content-Type: \(mimeType)\r\n\r\n".data(using: .utf8)!)
        body.append(fileData)
        body.append("\r\n".data(using: .utf8)!)
        body.append("--\(boundary)--\r\n".data(using: .utf8)!)
        return body
    }
    
    func sendVote(submissionId: Int) async throws -> Vote? {
        guard let url = URL(string: "http://localhost/submissions/\(submissionId)/vote") else {
            throw NetworkError.invalidURL
        }

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        
        let (data, response) = try await URLSession.shared.data(for: request)

        guard let httpResponse = response as? HTTPURLResponse, httpResponse.statusCode == 201 else {
            throw NetworkError.invalidResponse
        }
        
        do {
            let createdPost = try JSONDecoder().decode(Vote.self, from: data)
            return createdPost
        } catch {
            throw NetworkError.decodingError
        }
    }
    
    func testServerIsRunning() async throws -> Bool {
        guard let url = URL(string: "http://localhost:8585/") else {
            throw NetworkError.invalidURL
        }

        let (_, response) = try await URLSession.shared.data(from: url)

        guard let httpResponse = response as? HTTPURLResponse, httpResponse.statusCode == 200 else {
            print("Connection to server failed.")
            return false
        }
        
        return true
    }
    
    func createContest(name: String, description : String, startDate: String, endDate: String, votingEndDate: String) async throws -> Contest? {
        guard let url = URL(string: "http://localhost/contests") else {
            throw NetworkError.invalidURL
        }

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")

        let userData = try JSONSerialization.data(withJSONObject: ["name": name, "description": description, "submission_start_date": startDate, "submission_end_date" : endDate,
                                                                   "voting_end_date" : votingEndDate], options: [])

        request.httpBody = userData

        let (data, response) = try await URLSession.shared.data(for: request)
        print(String(data: data, encoding: .utf8) ?? "No data")

        guard let httpResponse = response as? HTTPURLResponse, httpResponse.statusCode == 201 else {
            throw NetworkError.invalidResponse
        }
        
        
        do {
            let createdPost = try JSONDecoder().decode(Contest.self, from: data)
            return createdPost
        } catch {
            throw NetworkError.decodingError
        }
    }
    
}

extension Data {
    mutating func append(_ string: String) {
        if let data = string.data(using: .utf8) {
            append(data)
        }
    }
}

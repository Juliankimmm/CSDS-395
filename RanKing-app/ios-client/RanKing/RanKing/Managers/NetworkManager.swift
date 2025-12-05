//
//  NetworkManager.swift
//  RanKing
//
//  Created by Jonathan Da Silva on 10/13/25.
//

import Foundation
import SwiftUI

enum NetworkError: Error {
    case invalidURL
    case invalidResponse
    case decodingError
}

@MainActor 
class NetworkManager: ObservableObject {
    
    private static var instance: NetworkManager?
    
    public static func getInstance() -> NetworkManager {
        if instance == nil {
            instance = NetworkManager()
        }
        return instance!
    }
        
    // MARK: - GET Request
    func fetchContests(query: String? = "") async throws -> [Contest]? {
        //TODO add query if necessary
        guard let url = URL(string: "https://b5xfrkkof2.execute-api.us-east-2.amazonaws.com/test/contests") else {
            throw NetworkError.invalidURL
        }
        let (data, response) = try await URLSession.shared.data(from: url)
        print(String(data: data, encoding: .utf8) ?? "No data")

        guard let httpResponse = response as? HTTPURLResponse, httpResponse.statusCode == 200 else {
            throw NetworkError.invalidResponse
        }
        
        let decoder = JSONDecoder()

        // setup date formatter
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy-MM-dd HH:mm:ss"
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.timeZone = TimeZone(secondsFromGMT: 0)

        decoder.dateDecodingStrategy = .formatted(formatter)
        
        print("200!")
        var contestsRes: [Contest]? = nil
        do {
            let response = try decoder.decode(ContestResponse.self, from: data)
            let contestsData = Data(response.body.utf8)
            contestsRes = try decoder.decode([Contest].self, from: contestsData)
        } catch {
            print("Decoding Error :( \(error)")
            throw NetworkError.decodingError
        }
        
        print("DECODED!!! \(contestsRes ?? [])")
        return contestsRes
    }
    
    func fetchSumbissions(contestId: Int) async throws -> [Submission]? {
        print(contestId)
        guard let url = URL(string: "https://b5xfrkkof2.execute-api.us-east-2.amazonaws.com/Deploy1/contests/\(contestId)/submissions") else {
            throw NetworkError.invalidURL
        }
        let (data, response) = try await URLSession.shared.data(from: url)
        print(String(data: data, encoding: .utf8) ?? "No data")
        guard let httpResponse = response as? HTTPURLResponse, httpResponse.statusCode == 200 else {
            throw NetworkError.invalidResponse
        }
        
        
        var submissions : [Submission]? = nil
        do {
            submissions = try JSONDecoder().decode([Submission].self, from: data)
        } catch {
            print("\(error)")
            throw NetworkError.decodingError
        }
        return submissions
    }

    // MARK: - POST Request
    func register(username: String, email: String, password: String) async throws -> Bool? {
        guard let url = URL(string: "https://b5xfrkkof2.execute-api.us-east-2.amazonaws.com/Deploy1/register") else {
            throw NetworkError.invalidURL
        }

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")

        let userData = try JSONSerialization.data(withJSONObject: ["username": username, "email": email, "password_hash": password], options: [])

        request.httpBody = userData

        let (data, response) = try await URLSession.shared.data(for: request)
        print(String(data: data, encoding: .utf8) ?? "No data")
        
        guard let httpResponse = response as? HTTPURLResponse, httpResponse.statusCode == 200 else {
            return false;
        }
        
        let result = String(data: data, encoding: .utf8) ?? "No Data"
        
        if (result.contains("Successfully")) {
            return true
        }
        return false
    }
    
    func login(email: String, password: String) async throws -> Bool? {
        print("login called")
        guard let url = URL(string: "https://b5xfrkkof2.execute-api.us-east-2.amazonaws.com/Deploy1/login") else {
            throw NetworkError.invalidURL
        }
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        let userData = try JSONSerialization.data(withJSONObject: ["email": email, "password_hash": password], options: [])
        request.httpBody = userData
        let (data, response) = try await URLSession.shared.data(for: request)

        print(String(data: data, encoding: .utf8) ?? "No data")
        guard let httpResponse = response as? HTTPURLResponse, httpResponse.statusCode == 200 else {
            return false
        }
        return true
    }
    
    func sendSubmission(imageData: Data, contestId: Int) async throws -> Bool {
        guard let url = URL(string: "https://b5xfrkkof2.execute-api.us-east-2.amazonaws.com/Deploy1/contests/\(contestId)/submissions") else {
            throw NetworkError.invalidURL
        }
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
//        request.setValue("Bearer \(accessToken)", forHTTPHeaderField: "Authorization")
        let boundary = "Boundary-\(UUID().uuidString)"
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
        request.httpBody = createMultipartBody(
            boundary: boundary,
            fieldName: "file",
            fileName: "submission.jpg",
            mimeType: "image/jpeg",
            fileData: imageData
        )
        let (data, response) = try await URLSession.shared.data(for: request)
        print(String(data: data, encoding: .utf8) ?? "No data")
        guard let httpResponse = response as? HTTPURLResponse, httpResponse.statusCode == 200 else {
            return false;
        }
        return true
    }
    
    
    func sendSubmission2(imageData: Data, contestId: Int, userId: Int) async throws -> Bool {
        print("Sending image")
        guard let url = URL(string: "https://b5xfrkkof2.execute-api.us-east-2.amazonaws.com/Deploy1/contests/\(contestId)/submissions") else {
            throw NetworkError.invalidURL
        }
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        let base64StringImage = imageData.base64EncodedString();
        let userData = try JSONSerialization.data(withJSONObject: ["user_id": userId, "image": base64StringImage, "filename": "example.jpg"], options: [])

        print("Json: \(userData)")
        request.httpBody = userData
        
        let (data, response) = try await URLSession.shared.data(for: request)
        print(String(data: data, encoding: .utf8) ?? "No data")
        guard let httpResponse = response as? HTTPURLResponse, httpResponse.statusCode == 200 else {
            return false;
        }
        return true
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

    
    func sendVote(submissionId: Int, userId : Int) async throws -> Bool? {
        guard let url = URL(string: "https://b5xfrkkof2.execute-api.us-east-2.amazonaws.com/Deploy1/submissions/\(submissionId)/vote") else {
            throw NetworkError.invalidURL
        }

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        let userData = try JSONSerialization.data(withJSONObject: ["user_id": userId], options: [])
        request.httpBody = userData
        
        let (data, response) = try await URLSession.shared.data(for: request)

        print(String(data: data, encoding: .utf8) ?? "No data")
        guard let httpResponse = response as? HTTPURLResponse, httpResponse.statusCode == 200 else {
            return false;
        }
        return true;
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
